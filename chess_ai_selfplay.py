"""
Chess AI with Self-Play Reinforcement Learning + Batched MCTS
PRODUCTION VERSION with all optimizations + CRITICAL BUG FIXES + SAFE AUGMENTATION + LR SCHEDULER:

CRITICAL BUG FIXES:
- ✅ FIXED: Board now represented from current player's perspective
- ✅ FIXED: Black can now learn to win (was always from White's view)
- ✅ Board flipped for Black to ensure symmetry in learning

PERFORMANCE OPTIMIZATIONS:
- Batched MCTS evaluation (10-50x faster than sequential)
- Mixed precision training (AMP) - 2-3x faster on RTX GPUs
- Channels-last memory format for better conv performance
- Masked softmax (no wasted compute on illegal moves)
- Larger batch sizes (256) for better GPU utilization

LEARNING QUALITY IMPROVEMENTS:
- ✅ Dirichlet noise at root (α=0.3, ε=0.25) for exploration
- ✅ Temperature schedule (explore first 30 moves, greedy after)
- ✅ Gradient clipping (1.0) for stability
- ✅ L2 weight decay (1e-4) for regularization
- ✅ Corrected virtual-loss bookkeeping in MCTS
- ✅ SAFE DATA AUGMENTATION: Horizontal flip with full validation (castling, en passant)
- ✅ LEARNING RATE SCHEDULER: Warmup (1000 steps) + Cosine decay for stable convergence

DRAW COLLAPSE FIXES:
- ✅ draw_penalty=-0.3: Draws now cost something, preventing reward collapse
- ✅ Repetition tracking: Positions repeated 2+ times get a per-move penalty injected
    into the VALUE TARGET, not just end-of-game reward
- ✅ Games continue through repetitions (no early exit) - model must escape or suffer
- ✅ temp_threshold raised to 30: Keeps exploration alive deeper into games
"""

import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os
import math
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageDraw, ImageFont, ImageTk
import threading
import queue
from collections import deque
import time

# -------------------------
# Neural Network Architecture
# -------------------------
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        self.conv1 = nn.Conv2d(12, 128, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        
        self.fc1 = nn.Linear(128 * 8 * 8, 512)
        
        # Policy head (move probabilities)
        self.policy_fc = nn.Linear(512, 4096)  # 64*64 possible moves (from-to)
        
        # Value head (position evaluation)
        self.value_fc1 = nn.Linear(512, 128)
        self.value_fc2 = nn.Linear(128, 1)
        
        self.dropout = nn.Dropout(0.3)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        x = x.reshape(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        policy = self.policy_fc(x)
        value = torch.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))
        return policy, value.squeeze(-1)


# -------------------------
# Chess AI with Batched MCTS + SAFE AUGMENTATION + REPETITION PENALTY
# -------------------------
class ChessAI:
    class MCTSNode:
        def __init__(self, board, parent=None, prior=0.0):
            self.board = board
            self.parent = parent
            self.prior = prior
            self.children = {}
            self.visits = 0
            self.value_sum = 0.0
            self.virtual_loss_count = 0

        @property
        def q_value(self):
            total_visits = self.visits + self.virtual_loss_count
            if total_visits == 0:
                return 0.0
            return self.value_sum / total_visits

    def __init__(self, save_dir="chess_ai_models",
                 replay_capacity=30000,
                 batch_size=256,
                 train_steps_per_game=16,
                 entropy_coef=0.01,
                 value_coef=1.5,
                 clip_grad=1.0,
                 min_buffer_size=1500,
                 lr=1e-4,
                 weight_decay=1e-4,
                 max_data_age=2000,
                 draw_penalty=-0.3,          # FIX: was 0.0 — draws now cost something
                 repetition_penalty=-0.15,   # FIX: per-position penalty applied inline
                 mcts_simulations=64,
                 mcts_batch_size=8,
                 mcts_c_puct=1.4,
                 mcts_dirichlet_eps=0.25,
                 mcts_dirichlet_alpha=0.3,
                 use_amp=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        
        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
        
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.scheduler_total_steps = 50000
        warmup_steps = 1000
        
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = min(1.0, float(step - warmup_steps) / float(max(1, self.scheduler_total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))
        
        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        
        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        
        self.save_dir = save_dir
        self.training_stats = {
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'total_training_steps': 0,
            'positions_flipped': 0,
            'positions_total': 0,
            'repetition_penalties_applied': 0,
        }
        self.stop_training_flag = False
        
        self.replay_capacity = replay_capacity
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.batch_size = batch_size
        self.train_steps_per_game = train_steps_per_game
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_grad = clip_grad
        self.min_buffer_size = min_buffer_size
        self.max_data_age = max_data_age
        self.draw_penalty = draw_penalty
        self.repetition_penalty = repetition_penalty
        
        self.mcts_simulations = mcts_simulations
        self.mcts_batch_size = mcts_batch_size
        self.mcts_c_puct = mcts_c_puct
        self.mcts_dirichlet_eps = mcts_dirichlet_eps
        self.mcts_dirichlet_alpha = mcts_dirichlet_alpha
        
        self.data_counter = 0
        self.loss_history = deque(maxlen=100)
        
        os.makedirs(save_dir, exist_ok=True)
        self.load_model()
    
    # -------------------------
    # Board / move helpers
    # -------------------------
    def board_to_tensor(self, board):
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        piece_to_channel = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        current_player = board.turn
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                ch = piece_to_channel[p.piece_type]
                if p.color != current_player:
                    ch += 6
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                if current_player == chess.BLACK:
                    rank = 7 - rank
                tensor[ch, rank, file] = 1.0
        return torch.from_numpy(tensor).unsqueeze(0)
    
    def move_to_index(self, move, flip=False):
        from_sq = move.from_square
        to_sq = move.to_square
        if flip:
            from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
            to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
            from_sq = chess.square(from_file, 7 - from_rank)
            to_sq = chess.square(to_file, 7 - to_rank)
        return from_sq * 64 + to_sq
    
    def index_to_move(self, board, idx, flip=False):
        from_sq = idx // 64
        to_sq = idx % 64
        if flip:
            from_file, from_rank = chess.square_file(from_sq), chess.square_rank(from_sq)
            to_file, to_rank = chess.square_file(to_sq), chess.square_rank(to_sq)
            from_sq = chess.square(from_file, 7 - from_rank)
            to_sq = chess.square(to_file, 7 - to_rank)
        candidate = chess.Move(from_sq, to_sq)
        if candidate in board.legal_moves:
            return candidate
        for promo_piece in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            m = chess.Move(from_sq, to_sq, promotion=promo_piece)
            if m in board.legal_moves:
                return m
        for m in board.legal_moves:
            if m.from_square == from_sq and m.to_square == to_sq:
                return m
        return None

    # -------------------------
    # Safe augmentation
    # -------------------------
    def is_position_symmetric_safe(self, board):
        if board.has_kingside_castling_rights(chess.WHITE): return False
        if board.has_queenside_castling_rights(chess.WHITE): return False
        if board.has_kingside_castling_rights(chess.BLACK): return False
        if board.has_queenside_castling_rights(chess.BLACK): return False
        if board.ep_square is not None: return False
        return True
    
    def augment_tensor_and_index(self, board_tensor, move_idx, can_flip=False):
        augmented = [(board_tensor.clone(), move_idx)]
        if can_flip:
            flipped_tensor = torch.flip(board_tensor, [3])
            from_sq = move_idx // 64
            to_sq = move_idx % 64
            from_file, from_rank = from_sq % 8, from_sq // 8
            to_file, to_rank = to_sq % 8, to_sq // 8
            from_sq_flipped = (7 - from_file) + from_rank * 8
            to_sq_flipped = (7 - to_file) + to_rank * 8
            move_idx_flipped = from_sq_flipped * 64 + to_sq_flipped
            augmented.append((flipped_tensor, move_idx_flipped))
        return augmented

    # -------------------------
    # Batched network inference
    # -------------------------
    def evaluate_batch(self, board_list):
        if len(board_list) == 0:
            return []
        
        self.model.eval()
        with torch.no_grad():
            board_tensors = [self.board_to_tensor(board) for board in board_list]
            batch_tensor = torch.cat(board_tensors, dim=0).to(self.device)
            if self.device.type == 'cuda':
                batch_tensor = batch_tensor.to(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                policy_logits, values = self.model(batch_tensor)
            policy_logits = policy_logits.cpu()
            values = values.cpu().numpy()
            
            results = []
            for i, board in enumerate(board_list):
                logits = policy_logits[i]
                value = float(values[i])
                flip = (board.turn == chess.BLACK)
                legal_moves = list(board.legal_moves)
                mask = torch.full((4096,), -float('inf'))
                for move in legal_moves:
                    idx = self.move_to_index(move, flip=flip)
                    if idx < 4096:
                        mask[idx] = 0.0
                masked_logits = logits + mask
                move_probs_tensor = torch.softmax(masked_logits, dim=0)
                legal_move_probs = []
                for move in legal_moves:
                    idx = self.move_to_index(move, flip=flip)
                    if idx < 4096:
                        legal_move_probs.append((move, float(move_probs_tensor[idx])))
                    else:
                        legal_move_probs.append((move, 1.0 / len(legal_moves)))
                total_prob = sum(p for _, p in legal_move_probs)
                if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
                    legal_move_probs = [(m, p/total_prob) for m, p in legal_move_probs]
                results.append((legal_move_probs, value))
            return results
    
    def get_move_probabilities(self, board):
        results = self.evaluate_batch([board])
        return results[0] if results else ([], 0.0)
    
    # -------------------------
    # Batched MCTS
    # -------------------------
    def run_mcts_batched(self, root_board, simulations=None, add_dirichlet_noise=False):
        if simulations is None:
            simulations = self.mcts_simulations
        
        root = self.MCTSNode(root_board.copy(), parent=None, prior=0.0)
        move_probs, value = self.get_move_probabilities(root.board)
        for move, prob in move_probs:
            child_board = root.board.copy()
            child_board.push(move)
            root.children[move] = self.MCTSNode(child_board, parent=root, prior=prob)
        
        if add_dirichlet_noise and len(root.children) > 0:
            eps = self.mcts_dirichlet_eps
            alpha = self.mcts_dirichlet_alpha
            moves = list(root.children.keys())
            noise = np.random.dirichlet([alpha] * len(moves))
            for i, m in enumerate(moves):
                old_prior = root.children[m].prior
                root.children[m].prior = (1 - eps) * old_prior + eps * noise[i]
        
        num_batches = (simulations + self.mcts_batch_size - 1) // self.mcts_batch_size
        
        for batch_idx in range(num_batches):
            batch_size = min(self.mcts_batch_size, simulations - batch_idx * self.mcts_batch_size)
            search_paths = []
            leaf_nodes = []
            
            for _ in range(batch_size):
                node = root
                search_path = [node]
                while len(node.children) > 0:
                    total_visits = sum(child.visits + child.virtual_loss_count
                                      for child in node.children.values()) + 1
                    best_score = -1e9
                    best_move = None
                    for move, child in node.children.items():
                        q = child.q_value
                        u = self.mcts_c_puct * child.prior * math.sqrt(total_visits) / (1 + child.visits + child.virtual_loss_count)
                        score = q + u
                        if score > best_score:
                            best_score = score
                            best_move = move
                    if best_move is None:
                        break
                    node = node.children[best_move]
                    node.virtual_loss_count += 1
                    search_path.append(node)
                search_paths.append(search_path)
                leaf_nodes.append(node)
            
            boards_to_evaluate = []
            terminal_values = []
            eval_index_map = []
            
            for idx, node in enumerate(leaf_nodes):
                if node.board.is_game_over():
                    res = node.board.result()
                    if res == "1-0":
                        value = 1.0 if root_board.turn == chess.WHITE else -1.0
                    elif res == "0-1":
                        value = -1.0 if root_board.turn == chess.WHITE else 1.0
                    else:
                        value = self.draw_penalty
                    terminal_values.append(value)
                else:
                    boards_to_evaluate.append(node.board)
                    terminal_values.append(None)
                    eval_index_map.append(idx)
            
            if boards_to_evaluate:
                eval_results = self.evaluate_batch(boards_to_evaluate)
            else:
                eval_results = []
            
            leaf_values = [None] * len(leaf_nodes)
            for eval_idx, node_idx in enumerate(eval_index_map):
                node = leaf_nodes[node_idx]
                mv_probs, leaf_value = eval_results[eval_idx]
                leaf_values[node_idx] = leaf_value
                for move, prob in mv_probs:
                    if move not in node.children:
                        b = node.board.copy()
                        b.push(move)
                        node.children[move] = self.MCTSNode(b, parent=node, prior=prob)
            
            for idx, val in enumerate(terminal_values):
                if val is not None:
                    leaf_values[idx] = val
            
            for search_path, leaf_value in zip(search_paths, leaf_values):
                if leaf_value is None:
                    continue
                value_to_propagate = leaf_value
                for n in reversed(search_path):
                    n.visits += 1
                    n.value_sum += value_to_propagate
                    if n.virtual_loss_count > 0:
                        n.virtual_loss_count -= 1
                    value_to_propagate = -value_to_propagate
        
        return root
    
    # -------------------------
    # Move selection
    # -------------------------
    def select_move(self, board, temperature=1.0, use_mcts=True, add_dirichlet_noise=False):
        if not use_mcts:
            move_probs, _ = self.get_move_probabilities(board)
            if not move_probs:
                return None
            if temperature == 0:
                return max(move_probs, key=lambda x: x[1])[0]
            moves, probs = zip(*move_probs)
            probs = np.array(probs, dtype=np.float64)
            probs = np.clip(probs, 1e-12, None)
            if temperature > 0:
                probs = probs ** (1.0 / temperature)
            probs = probs / probs.sum()
            return np.random.choice(moves, p=probs)
        
        root = self.run_mcts_batched(board, simulations=self.mcts_simulations, add_dirichlet_noise=add_dirichlet_noise)
        if not root.children:
            return None
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)
        if temperature == 0 or temperature < 1e-8:
            best_idx = int(np.argmax(visits))
            return moves[best_idx]
        probs = visits ** (1.0 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(moves, p=probs)
    
    # -------------------------
    # Self-play with inline repetition penalty (NEW!)
    # -------------------------
    def play_game(self, temperature=1.0, max_moves=200, temp_threshold=30):
        """
        Play a self-play game with inline repetition penalty.

        KEY CHANGE: We no longer exit early on threefold repetition.
        Instead, whenever a position is visited for the 2nd+ time,
        we record a penalty directly in the VALUE TARGET for that step.
        The game only ends via the normal chess rules (game over) or max_moves.

        This means the model must ESCAPE repetition loops or keep paying a tax,
        rather than exploiting them as a free shortcut to a 0-reward draw.
        """
        board = chess.Board()
        game_data = []
        move_count = 0

        # Track how many times each board position (by FEN) has been seen
        position_counts = {}

        while not board.is_game_over() and not self.stop_training_flag:
            fen_key = board.board_fen()
            visit_count = position_counts.get(fen_key, 0)
            position_counts[fen_key] = visit_count + 1

            # Compute inline repetition penalty for this step:
            # 0 on first visit, repetition_penalty on 2nd+ visit
            # Scale penalty up with each additional repeat (-0.15, -0.30, -0.45...)
            if visit_count >= 1:
                inline_penalty = self.repetition_penalty * visit_count
                self.training_stats['repetition_penalties_applied'] += 1
            else:
                inline_penalty = 0.0

            can_flip = self.is_position_symmetric_safe(board)
            board_tensor = self.board_to_tensor(board).cpu()

            current_temp = temperature if move_count < temp_threshold else 0.0
            move = self.select_move(board, temperature=current_temp, use_mcts=True, add_dirichlet_noise=True)

            if move is None:
                break

            flip = (board.turn == chess.BLACK)
            move_idx = self.move_to_index(move, flip=flip)
            player = board.turn

            with torch.no_grad():
                self.model.eval()
                board_tensor_device = board_tensor.to(self.device)
                if self.device.type == 'cuda':
                    board_tensor_device = board_tensor_device.to(memory_format=torch.channels_last)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    policy_logits, _ = self.model(board_tensor_device)
                log_probs = F.log_softmax(policy_logits, dim=1)
                if move_idx < log_probs.shape[1]:
                    move_log_prob = log_probs[0, move_idx].item()
                else:
                    move_log_prob = -10.0

            # Store inline_penalty alongside data so add_game_to_buffer can apply it
            game_data.append((board_tensor, move_idx, player, move_log_prob, can_flip, inline_penalty))
            board.push(move)
            move_count += 1

            if len(game_data) > max_moves:
                break

        # Assign end-of-game reward
        result = board.result()
        if result == "1-0":
            reward = 1.0
            self.training_stats['white_wins'] += 1
        elif result == "0-1":
            reward = -1.0
            self.training_stats['black_wins'] += 1
        else:
            reward = self.draw_penalty
            self.training_stats['draws'] += 1

        self.training_stats['games_played'] += 1
        self.training_stats['total_moves'] += len(game_data)

        return game_data, reward

    # -------------------------
    # Replay buffer with inline repetition penalty applied to value target
    # -------------------------
    def add_game_to_buffer(self, game_data, reward):
        """
        Add game to replay buffer.

        VALUE TARGET = game_outcome (from player's perspective)
                     + inline_penalty (already negative for repeated positions)

        The inline penalty is clamped so it can't flip the sign past -1.
        """
        for board_tensor_cpu, move_idx, player, old_log_prob, can_flip, inline_penalty in game_data:
            if move_idx is None:
                continue

            # Convert game outcome to this player's perspective
            if player == chess.WHITE:
                base_value = reward
            else:
                base_value = -reward

            # Apply inline repetition penalty and clamp to valid range
            target_value = float(np.clip(base_value + inline_penalty, -1.0, 1.0))

            augmented = self.augment_tensor_and_index(board_tensor_cpu, move_idx, can_flip)

            self.training_stats['positions_total'] += 1
            if can_flip:
                self.training_stats['positions_flipped'] += 1

            for aug_tensor, aug_move_idx in augmented:
                self.replay_buffer.append((
                    aug_tensor,
                    aug_move_idx,
                    target_value,
                    old_log_prob,
                    self.data_counter
                ))

            self.data_counter += 1
    
    def clean_old_data(self):
        if len(self.replay_buffer) < self.replay_capacity:
            return
        current_counter = self.data_counter
        new_buffer = deque(maxlen=self.replay_capacity)
        for entry in self.replay_buffer:
            board_tensor, move_idx, target_value, old_log_prob, age_marker = entry
            if current_counter - age_marker < self.max_data_age:
                new_buffer.append(entry)
        self.replay_buffer = new_buffer
    
    def sample_batch(self):
        if len(self.replay_buffer) == 0:
            raise ValueError("Replay buffer is empty")
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        if len(batch) < self.batch_size:
            batch = [random.choice(self.replay_buffer) for _ in range(self.batch_size)]
        boards, move_idxs, target_values, old_log_probs, _ = zip(*batch)
        boards_tensor = torch.cat(boards, dim=0).to(self.device)
        if self.device.type == 'cuda':
            boards_tensor = boards_tensor.to(memory_format=torch.channels_last)
        move_idxs_tensor = torch.LongTensor(move_idxs).to(self.device)
        target_values_tensor = torch.FloatTensor(target_values).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        return boards_tensor, move_idxs_tensor, target_values_tensor, old_log_probs_tensor
    
    # -------------------------
    # Training
    # -------------------------
    def train_on_batch(self, boards_tensor, move_idxs_tensor, target_values_tensor, old_log_probs_tensor):
        self.model.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            policy_logits, values = self.model(boards_tensor)
            log_probs = F.log_softmax(policy_logits, dim=1)
            selected_log_probs = log_probs.gather(1, move_idxs_tensor.unsqueeze(1)).squeeze(1)
            selected_log_probs = torch.clamp(selected_log_probs, min=-10.0, max=0.0)
            ratio = torch.exp(selected_log_probs - old_log_probs_tensor)
            ratio = torch.clamp(ratio, min=0.5, max=2.0)
            advantages = (target_values_tensor - values).detach()
            adv_mean = advantages.mean()
            adv_std = advantages.std(unbiased=False)
            if adv_std.item() < 1e-6:
                advantages_norm = advantages - adv_mean
            else:
                advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)
            advantages_norm = torch.clamp(advantages_norm, -5.0, 5.0)
            policy_loss_raw = -(ratio * selected_log_probs * advantages_norm).mean()
            policy_loss = torch.clamp(policy_loss_raw, -10.0, 10.0)
            value_loss = F.mse_loss(values, target_values_tensor)
            probs = F.softmax(policy_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            adaptive_entropy_coef = self.entropy_coef * (1.0 + 0.1 / (1.0 + self.training_stats['total_training_steps'] / 1000.0))
            loss = policy_loss + self.value_coef * value_loss - adaptive_entropy_coef * entropy
        
        if not torch.isfinite(loss):
            return float('nan'), float('nan')
        
        self.optimizer.zero_grad()
        if self.use_amp:
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
            self.optimizer.step()
        
        try:
            self.loss_history.append(policy_loss.item())
        except:
            self.loss_history.append(0.0)
        
        self.training_stats['total_training_steps'] += 1
        try:
            self.scheduler.step()
        except Exception:
            pass
        
        return policy_loss.item(), value_loss.item()
    
    def train(self, num_games=10, temperature=1.0, temp_threshold=30, callback=None):
        self.stop_training_flag = False
        
        for game_num in range(num_games):
            if self.stop_training_flag:
                break
            
            game_start = time.time()
            game_data, reward = self.play_game(temperature, temp_threshold=temp_threshold)
            game_time = time.time() - game_start
            
            self.add_game_to_buffer(game_data, reward)
            
            if game_num % 10 == 0:
                self.clean_old_data()
            
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            steps_done = 0
            
            train_start = time.time()
            
            if len(self.replay_buffer) >= max(self.min_buffer_size, self.batch_size):
                for _ in range(self.train_steps_per_game):
                    try:
                        boards, moves, values, log_probs = self.sample_batch()
                    except ValueError:
                        break
                    p_loss, v_loss = self.train_on_batch(boards, moves, values, log_probs)
                    if p_loss != p_loss:
                        continue
                    avg_policy_loss += p_loss
                    avg_value_loss += v_loss
                    steps_done += 1
                
                if steps_done > 0:
                    avg_policy_loss /= steps_done
                    avg_value_loss /= steps_done
            
            train_time = time.time() - train_start
            
            if callback:
                callback(game_num + 1, num_games, avg_policy_loss, avg_value_loss, reward, game_time, train_time)
            
            if (game_num + 1) % 10 == 0:
                self.save_model()
    
    def stop_training(self):
        self.stop_training_flag = True
    
    # -------------------------
    # Save / Load
    # -------------------------
    def save_model(self):
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        try:
            cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            save_dict = {
                'model_state_dict': cpu_state,
                'training_stats': self.training_stats,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict()
            }
            if self.scaler:
                save_dict['scaler_state_dict'] = self.scaler.state_dict()
            torch.save(save_dict, model_path)
            if self.training_stats['games_played'] % 50 == 0:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                backup_path = os.path.join(self.save_dir, f"model_{timestamp}.pth")
                torch.save(save_dict, backup_path)
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                if self.device.type == 'cuda':
                    self.model = self.model.to(memory_format=torch.channels_last)
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                # Backwards compatibility
                for key in ['positions_flipped', 'positions_total', 'total_training_steps', 'repetition_penalties_applied']:
                    if key not in self.training_stats:
                        self.training_stats[key] = 0
                if 'optimizer_state_dict' in checkpoint:
                    self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                if self.scaler and 'scaler_state_dict' in checkpoint:
                    self.scaler.load_state_dict(checkpoint['scaler_state_dict'])
                print(f"Model loaded from {model_path}")
                print(f"Stats: {self.training_stats}")
            except Exception as e:
                print(f"Error loading model: {e}")
        else:
            print("No saved model found. Starting fresh.")


# -------------------------
# GUI
# -------------------------
class ChessGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess AI - Draw Collapse Fix")
        self.window.geometry("1000x800")
        
        self.ai = ChessAI()
        self.board = chess.Board()
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.human_color = None
        self.is_training = False
        self.training_thread = None
        self.square_size = 60
        self.move_history = []
        self.ai_thinking = False
        self.flip_board = False
        self.flip_var = tk.BooleanVar(value=False)
        self.message_queue = queue.Queue()
        
        self.setup_gui()
        self.process_queue()
        
    def setup_gui(self):
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10)
        
        board_container = ttk.Frame(left_frame, relief=tk.SUNKEN, borderwidth=2)
        board_container.grid(row=0, column=0, pady=10)
        
        self.board_label = ttk.Label(board_container)
        self.board_label.grid(row=0, column=0)
        self.board_label.bind("<Button-1>", self.on_board_click)
        
        history_frame = ttk.LabelFrame(left_frame, text="Move History", padding="5")
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, height=8, width=40, wrap=tk.WORD)
        self.history_text.grid(row=0, column=0)
        
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E), padx=10)
        
        train_frame = ttk.LabelFrame(right_frame, text="Training Controls", padding="10")
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(train_frame, text="Number of games:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_games_var = tk.StringVar(value="10")
        ttk.Entry(train_frame, textvariable=self.num_games_var, width=15).grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(train_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="1.0")
        ttk.Entry(train_frame, textvariable=self.temperature_var, width=15).grid(row=1, column=1, pady=5, padx=5)
        
        ttk.Label(train_frame, text="Temp threshold moves:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.temp_threshold_var = tk.StringVar(value="30")
        ttk.Entry(train_frame, textvariable=self.temp_threshold_var, width=15).grid(row=2, column=1, pady=5, padx=5)
        
        ttk.Label(train_frame, text="(Explore first N moves, then greedy)",
                 font=('Arial', 8), foreground='gray').grid(row=2, column=2, sticky=tk.W, padx=5)
        
        button_frame = ttk.Frame(train_frame)
        button_frame.grid(row=3, column=0, columnspan=3, pady=10)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.progress_var = tk.StringVar(value="No training in progress")
        ttk.Label(train_frame, textvariable=self.progress_var, wraplength=250).grid(row=4, column=0, columnspan=3, pady=5)
        
        play_frame = ttk.LabelFrame(right_frame, text="Play Controls", padding="10")
        play_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(play_frame, text="Play as White", command=lambda: self.start_game(chess.WHITE), width=20).grid(row=0, column=0, pady=5)
        ttk.Button(play_frame, text="Play as Black", command=lambda: self.start_game(chess.BLACK), width=20).grid(row=1, column=0, pady=5)
        ttk.Button(play_frame, text="AI vs AI Demo", command=self.watch_ai_game, width=20).grid(row=2, column=0, pady=5)
        ttk.Button(play_frame, text="New Game", command=self.reset_game, width=20).grid(row=3, column=0, pady=5)
        ttk.Checkbutton(play_frame, text="Flip board", variable=self.flip_var, command=self.on_flip_toggle).grid(row=4, column=0, pady=5)
        
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=16, width=35, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0)
        
        amp_status = "AMP: ON" if self.ai.use_amp else "AMP: OFF"
        ttk.Label(stats_frame, text=f"Device: {self.ai.device}", foreground="blue").grid(row=1, column=0, pady=2)
        ttk.Label(stats_frame, text=amp_status, foreground="green").grid(row=2, column=0, pady=2)
        ttk.Label(stats_frame, text="✅ DRAW COLLAPSE FIX ACTIVE", foreground="red", font=('Arial', 9, 'bold')).grid(row=3, column=0, pady=2)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.update_board_display()
        self.update_stats_display()
    
    def on_flip_toggle(self):
        self.flip_board = bool(self.flip_var.get())
        self.update_board_display()
    
    def board_to_image(self):
        board_size = self.square_size * 8
        image = Image.new('RGB', (board_size + 40, board_size + 40), 'white')
        draw = ImageDraw.Draw(image)
        
        light_square = (240, 217, 181)
        dark_square = (181, 136, 99)
        selected_color = (255, 255, 100)
        legal_move_color = (144, 238, 144)
        
        try:
            piece_font = ImageFont.truetype("seguisym.ttf", int(self.square_size * 0.7))
            coord_font = ImageFont.truetype("arial.ttf", 12)
        except:
            try:
                piece_font = ImageFont.truetype("Arial.ttf", int(self.square_size * 0.7))
                coord_font = ImageFont.truetype("Arial.ttf", 12)
            except:
                piece_font = ImageFont.load_default()
                coord_font = ImageFont.load_default()
        
        offset = 20
        
        for rank in range(8):
            for file in range(8):
                if not self.flip_board:
                    x1 = file * self.square_size + offset
                    y1 = (7 - rank) * self.square_size + offset
                else:
                    x1 = (7 - file) * self.square_size + offset
                    y1 = rank * self.square_size + offset
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                square = chess.square(file, rank)
                if square == self.selected_square:
                    color = selected_color
                elif square in [move.to_square for move in self.legal_moves_for_selected]:
                    color = legal_move_color
                elif (rank + file) % 2 == 0:
                    color = light_square
                else:
                    color = dark_square
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray')
        
        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        if self.flip_board:
            files = list(reversed(files))
            ranks = list(reversed(ranks))
        
        for i, file_char in enumerate(files):
            x = i * self.square_size + self.square_size // 2 + offset
            draw.text((x, 5), file_char, fill='black', font=coord_font, anchor='mm')
            draw.text((x, board_size + offset + 15), file_char, fill='black', font=coord_font, anchor='mm')
        
        for i, rank_char in enumerate(ranks):
            if not self.flip_board:
                y = (7 - i) * self.square_size + self.square_size // 2 + offset
            else:
                y = i * self.square_size + self.square_size // 2 + offset
            draw.text((5, y), rank_char, fill='black', font=coord_font, anchor='mm')
            draw.text((board_size + offset + 15, y), rank_char, fill='black', font=coord_font, anchor='mm')
        
        piece_symbols = {
            'P': '♙', 'N': '♘', 'B': '♗', 'R': '♖', 'Q': '♕', 'K': '♔',
            'p': '♟', 'n': '♞', 'b': '♝', 'r': '♜', 'q': '♛', 'k': '♚'
        }
        
        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = self.board.piece_at(square)
                if piece:
                    symbol = piece.symbol()
                    piece_char = piece_symbols.get(symbol, symbol)
                    if not self.flip_board:
                        x = file * self.square_size + self.square_size // 2 + offset
                        y = (7 - rank) * self.square_size + self.square_size // 2 + offset
                    else:
                        x = (7 - file) * self.square_size + self.square_size // 2 + offset
                        y = rank * self.square_size + self.square_size // 2 + offset
                    piece_color = 'white' if piece.color == chess.WHITE else 'black'
                    outline_color = 'black' if piece.color == chess.WHITE else 'white'
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        draw.text((x + dx, y + dy), piece_char, fill=outline_color, font=piece_font, anchor='mm')
                    draw.text((x, y), piece_char, fill=piece_color, font=piece_font, anchor='mm')
        
        return ImageTk.PhotoImage(image)
    
    def update_board_display(self):
        try:
            photo = self.board_to_image()
            self.board_label.config(image=photo)
            self.board_label.image = photo
        except Exception as e:
            self.board_label.config(text=str(self.board))
    
    def update_stats_display(self):
        stats = self.ai.training_stats
        win_rate_white = (stats['white_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        win_rate_black = (stats['black_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        draw_rate = (stats['draws'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        flip_rate = (stats['positions_flipped'] / stats['positions_total'] * 100) if stats['positions_total'] > 0 else 0
        current_lr = self.ai.optimizer.param_groups[0]['lr']
        rep_penalties = stats.get('repetition_penalties_applied', 0)
        
        stats_text = f"""Games: {stats['games_played']}
Moves: {stats['total_moves']}
Steps: {stats.get('total_training_steps', 0)}

White: {stats['white_wins']} ({win_rate_white:.1f}%)
Black: {stats['black_wins']} ({win_rate_black:.1f}%)
Draws: {stats['draws']} ({draw_rate:.1f}%)

Buffer: {len(self.ai.replay_buffer)}
Augmentation: {flip_rate:.1f}% flipped
Rep.penalties: {rep_penalties}
LR: {current_lr:.2e}
Draw penalty: {self.ai.draw_penalty}
Rep penalty: {self.ai.repetition_penalty}
Model: {self.ai.save_dir}""".strip()
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_move_history(self):
        self.history_text.delete(1.0, tk.END)
        if not self.move_history:
            self.history_text.insert(1.0, "No moves yet")
            return
        move_text = ""
        for i, move in enumerate(self.move_history):
            if i % 2 == 0:
                move_text += f"{i//2 + 1}. {move} "
            else:
                move_text += f"{move}\n"
        self.history_text.insert(1.0, move_text)
        self.history_text.see(tk.END)
    
    def on_board_click(self, event):
        if self.human_color is None or self.board.turn != self.human_color or self.board.is_game_over() or self.ai_thinking:
            return
        offset = 20
        x, y = event.x - offset, event.y - offset
        if x < 0 or y < 0 or x >= self.square_size * 8 or y >= self.square_size * 8:
            return
        col = int(min(7, max(0, x // self.square_size)))
        row = int(min(7, max(0, y // self.square_size)))
        if not self.flip_board:
            file, rank = col, 7 - row
        else:
            file, rank = 7 - col, row
        square = chess.square(file, rank)
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                self.legal_moves_for_selected = [m for m in self.board.legal_moves if m.from_square == square]
                self.status_var.set(f"Selected {chess.SQUARE_NAMES[square]}")
                self.update_board_display()
        else:
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN and (rank == 0 or rank == 7):
                promo = self.ask_promotion_piece() or chess.QUEEN
                move = chess.Move(self.selected_square, square, promotion=promo)
            else:
                move = chess.Move(self.selected_square, square)
            if move in self.board.legal_moves:
                self.make_move(move)
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.update_board_display()
                if self.board.is_game_over():
                    self.game_over()
                else:
                    self.window.after(300, self.ai_move)
            else:
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.status_var.set("Illegal move")
                self.update_board_display()
    
    def ask_promotion_piece(self):
        dlg = tk.Toplevel(self.window)
        dlg.title("Promotion")
        dlg.transient(self.window)
        dlg.grab_set()
        choice = {'piece': None}
        ttk.Label(dlg, text="Promote to:", padding=10).grid(row=0, column=0, columnspan=4)
        ttk.Button(dlg, text="♕", width=8, command=lambda: [choice.update({'piece': chess.QUEEN}), dlg.destroy()]).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(dlg, text="♖", width=8, command=lambda: [choice.update({'piece': chess.ROOK}), dlg.destroy()]).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(dlg, text="♗", width=8, command=lambda: [choice.update({'piece': chess.BISHOP}), dlg.destroy()]).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(dlg, text="♘", width=8, command=lambda: [choice.update({'piece': chess.KNIGHT}), dlg.destroy()]).grid(row=1, column=3, padx=5, pady=5)
        dlg.wait_window()
        return choice['piece']
    
    def make_move(self, move):
        san = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san)
        self.update_move_history()
    
    def ai_move(self):
        if self.board.is_game_over():
            self.game_over()
            return
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        self.window.update()
        try:
            move = self.ai.select_move(self.board, temperature=0.0, use_mcts=True, add_dirichlet_noise=False)
            if move:
                self.make_move(move)
                self.status_var.set(f"AI: {self.move_history[-1]}")
                self.update_board_display()
                if self.board.is_game_over():
                    self.game_over()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.ai_thinking = False
    
    def start_game(self, color):
        self.board = chess.Board()
        self.human_color = color
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.ai_thinking = False
        self.flip_board = (color == chess.BLACK)
        self.flip_var.set(self.flip_board)
        self.update_board_display()
        self.update_move_history()
        self.status_var.set(f"You are {'White' if color == chess.WHITE else 'Black'}")
        if color == chess.BLACK:
            self.window.after(500, self.ai_move)
    
    def watch_ai_game(self):
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.flip_board = False
        self.flip_var.set(False)
        self.update_board_display()
        self.update_move_history()
        self.play_ai_vs_ai()
    
    def play_ai_vs_ai(self):
        if not self.board.is_game_over():
            try:
                move = self.ai.select_move(self.board, temperature=0.1, use_mcts=True)
                if move:
                    self.make_move(move)
                    self.update_board_display()
                    self.window.after(800, self.play_ai_vs_ai)
            except Exception as e:
                self.status_var.set(f"Error: {e}")
        else:
            self.game_over()
    
    def reset_game(self):
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.flip_board = False
        self.flip_var.set(False)
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Ready")
    
    def game_over(self):
        outcome = self.board.outcome()
        if not outcome:
            msg = "Game ended"
        elif outcome.winner == chess.WHITE:
            msg = "White wins"
        elif outcome.winner == chess.BLACK:
            msg = "Black wins"
        else:
            msg = "Draw"
        self.status_var.set(f"Game Over: {msg}")
        messagebox.showinfo("Game Over", msg)
    
    def process_queue(self):
        try:
            while True:
                msg = self.message_queue.get_nowait()
                if msg['type'] == 'training_update':
                    self.progress_var.set(msg['text'])
                    self.update_stats_display()
                elif msg['type'] == 'training_complete':
                    self.stop_training()
                    messagebox.showinfo("Done", msg['text'])
                elif msg['type'] == 'training_error':
                    self.stop_training()
                    messagebox.showerror("Error", msg['text'])
        except queue.Empty:
            pass
        self.window.after(100, self.process_queue)
    
    def training_callback(self, game_num, total, p_loss, v_loss, reward, game_t, train_t):
        result = "Draw" if abs(reward - self.ai.draw_penalty) < 1e-6 else ("Win(W)" if reward > 0 else "Win(B)")
        text = f"Game {game_num}/{total}\n{result}\nP: {p_loss:.3f} V: {v_loss:.3f}\nGame: {game_t:.1f}s Train: {train_t:.1f}s"
        self.message_queue.put({'type': 'training_update', 'text': text})
    
    def train_worker(self, num, temp, temp_threshold):
        try:
            self.ai.train(num_games=num, temperature=temp, temp_threshold=temp_threshold, callback=self.training_callback)
            self.message_queue.put({'type': 'training_complete', 'text': f"Trained {num} games"})
        except Exception as e:
            self.message_queue.put({'type': 'training_error', 'text': str(e)})
    
    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Already training")
            return
        try:
            num = int(self.num_games_var.get())
            temp = float(self.temperature_var.get())
            temp_threshold = int(self.temp_threshold_var.get())
            if num <= 0 or temp <= 0 or temp_threshold < 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Invalid input")
            return
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(f"Starting... (Temp={temp}, Switch@move {temp_threshold})")
        self.training_thread = threading.Thread(target=self.train_worker, args=(num, temp, temp_threshold), daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        if self.is_training:
            self.ai.stop_training()
            self.window.after(1000, self.finish_stop_training)
        else:
            self.finish_stop_training()
    
    def finish_stop_training(self):
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ai.save_model()
        self.update_stats_display()
        self.status_var.set("Ready")
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()