"""
Chess AI with Self-Play Reinforcement Learning + Batched MCTS
PRODUCTION VERSION — ResNet architecture + improved augmentation

NETWORK IMPROVEMENTS:
- Residual blocks with skip connections (6 blocks, 128 channels)
- Batch Normalization after every conv layer
- AlphaZero-style policy head (conv → BN → ReLU → FC)
- AlphaZero-style value head (conv → BN → ReLU → FC → FC → tanh)
- Removed Dropout (replaced by BatchNorm regularization)

AUGMENTATION IMPROVEMENTS:
- Horizontal flip valid for ALL moves except castling (~99% of positions)
- En passant correctly included (pawn+target both flip cleanly)
- Castling excluded (king destination no longer valid after file flip)

CRITICAL BUG FIXES:
- Board represented from current player's perspective
- Black can learn to win (board flipped for Black)

PERFORMANCE:
- Batched MCTS evaluation
- Mixed precision training (AMP)
- Channels-last memory format
- Masked softmax on illegal moves

LEARNING:
- Dirichlet noise at root
- Temperature schedule
- Gradient clipping + L2 weight decay
- Safe horizontal flip augmentation
- LR warmup + cosine decay

DRAW COLLAPSE FIXES:
- draw_penalty=-0.3
- Repetition tracking with per-move value penalty

MCTS:
- Tree reuse between moves
- FPU (First Play Urgency)
- PUCT perspective fix
- Clean virtual loss
"""

import chess
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
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
# Residual Block
# -------------------------
class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        for m in [self.conv1, self.conv2]:
            nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


# -------------------------
# Neural Network
# -------------------------
class ChessNet(nn.Module):
    def __init__(self, channels=128, num_blocks=6):
        super().__init__()
        self.input_conv = nn.Conv2d(12, channels, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(channels)
        self.tower      = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])

        # Policy head
        self.policy_conv = nn.Conv2d(channels, 2, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(2)
        self.policy_fc   = nn.Linear(2 * 64, 4096)

        # Value head
        self.value_conv  = nn.Conv2d(channels, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(64, 128)
        self.value_fc2   = nn.Linear(128, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.tower(x)

        p = F.relu(self.policy_bn(self.policy_conv(x))).flatten(1)
        p = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x))).flatten(1)
        v = F.relu(self.value_fc1(v))
        v = torch.tanh(self.value_fc2(v)).squeeze(-1)
        return p, v


# -------------------------
# Chess AI
# -------------------------
class ChessAI:
    class MCTSNode:
        __slots__ = ('board', 'parent', 'prior', 'children', 'visits',
                     'value_sum', 'virtual_loss_count', 'turn')

        def __init__(self, board, parent=None, prior=0.0):
            self.board = board
            self.parent = parent
            self.prior = prior
            self.children = {}
            self.visits = 0
            self.value_sum = 0.0
            self.virtual_loss_count = 0
            self.turn = board.turn

        @property
        def q_value(self):
            if self.visits == 0:
                return 0.0
            return self.value_sum / self.visits

    def __init__(self, save_dir="chess_ai_models",
                 replay_capacity=30000,
                 batch_size=128,
                 train_steps_per_game=16,
                 entropy_coef=0.01,
                 value_coef=1.5,
                 clip_grad=1.0,
                 min_buffer_size=200,
                 lr=1e-4,
                 weight_decay=1e-4,
                 max_data_age=2000,
                 draw_penalty=-0.3,
                 repetition_penalty=-0.15,
                 mcts_simulations=128, # Change accordingly
                 mcts_batch_size=8,
                 mcts_c_puct=1.4,
                 mcts_dirichlet_eps=0.25,
                 mcts_dirichlet_alpha=0.3,
                 mcts_fpu_reduction=0.2,
                 net_channels=128,
                 net_blocks=6,
                 use_amp=True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet(channels=net_channels, num_blocks=net_blocks).to(self.device)

        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)

        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)

        self.scheduler_total_steps = 50000
        warmup_steps = 1000

        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = min(1.0, float(step - warmup_steps) /
                          float(max(1, self.scheduler_total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)

        self.use_amp = use_amp and self.device.type == 'cuda'
        self.scaler  = torch.cuda.amp.GradScaler() if self.use_amp else None

        self.save_dir = save_dir
        self.training_stats = {
            'games_played': 0, 'total_moves': 0,
            'white_wins': 0, 'black_wins': 0, 'draws': 0,
            'total_training_steps': 0,
            'positions_flipped': 0, 'positions_total': 0,
            'repetition_penalties_applied': 0,
        }
        self.stop_training_flag = False

        self.replay_capacity       = replay_capacity
        self.replay_buffer         = deque(maxlen=replay_capacity)
        self.batch_size            = batch_size
        self.train_steps_per_game  = train_steps_per_game
        self.entropy_coef          = entropy_coef
        self.value_coef            = value_coef
        self.clip_grad             = clip_grad
        self.min_buffer_size       = min_buffer_size
        self.max_data_age          = max_data_age
        self.draw_penalty          = draw_penalty
        self.repetition_penalty    = repetition_penalty

        self.mcts_simulations      = mcts_simulations
        self.mcts_batch_size       = mcts_batch_size
        self.mcts_c_puct           = mcts_c_puct
        self.mcts_dirichlet_eps    = mcts_dirichlet_eps
        self.mcts_dirichlet_alpha  = mcts_dirichlet_alpha
        self.mcts_fpu_reduction    = mcts_fpu_reduction

        self._mcts_root_cache = None
        self.data_counter     = 0
        self.loss_history     = deque(maxlen=100)

        os.makedirs(save_dir, exist_ok=True)
        self.load_model()

    def reset_mcts_tree(self):
        self._mcts_root_cache = None

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
        to_sq   = move.to_square
        if flip:
            from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
            to_sq   = chess.square(chess.square_file(to_sq),   7 - chess.square_rank(to_sq))
        return from_sq * 64 + to_sq

    def index_to_move(self, board, idx, flip=False):
        from_sq = idx // 64
        to_sq   = idx % 64
        if flip:
            from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
            to_sq   = chess.square(chess.square_file(to_sq),   7 - chess.square_rank(to_sq))
        candidate = chess.Move(from_sq, to_sq)
        if candidate in board.legal_moves:
            return candidate
        for promo in [chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT]:
            m = chess.Move(from_sq, to_sq, promotion=promo)
            if m in board.legal_moves:
                return m
        for m in board.legal_moves:
            if m.from_square == from_sq and m.to_square == to_sq:
                return m
        return None

    # -------------------------
    # Augmentation
    # -------------------------
    def augment_tensor_and_index(self, board_tensor, move_idx, can_flip=False):
        """
        Horizontal (file) flip augmentation.

        Safe for ALL moves EXCEPT castling:
        - Regular moves / captures: from/to squares flip correctly.
        - En passant: pawn and target both mirror cleanly.
        - Promotions: pawn on flipped file promotes on flipped file — fine.
        - Castling: king destination after file-flip is not a real castling
          square, so these are excluded by passing can_flip=False.

        The board tensor encodes only piece positions (12 channels), not
        castling rights or ep squares, so flipping is always layout-valid.
        """
        augmented = [(board_tensor.clone(), move_idx)]
        if can_flip:
            flipped_tensor = torch.flip(board_tensor, [3])
            from_sq   = move_idx // 64
            to_sq     = move_idx % 64
            # flip file, keep rank: new_sq = (7 - file) + rank * 8
            from_sq_f = (7 - from_sq % 8) + (from_sq // 8) * 8
            to_sq_f   = (7 - to_sq   % 8) + (to_sq   // 8) * 8
            augmented.append((flipped_tensor, from_sq_f * 64 + to_sq_f))
        return augmented

    # -------------------------
    # Batched inference
    # -------------------------
    def evaluate_batch(self, board_list):
        if not board_list:
            return []
        self.model.eval()
        with torch.no_grad():
            batch = torch.cat([self.board_to_tensor(b) for b in board_list], dim=0).to(self.device)
            if self.device.type == 'cuda':
                batch = batch.to(memory_format=torch.channels_last)
            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                policy_logits, values = self.model(batch)
            policy_logits = policy_logits.cpu()
            values        = values.cpu().numpy()

            results = []
            for i, board in enumerate(board_list):
                logits      = policy_logits[i]
                value       = float(values[i])
                flip        = (board.turn == chess.BLACK)
                legal_moves = list(board.legal_moves)
                mask        = torch.full((4096,), -float('inf'))
                for move in legal_moves:
                    idx = self.move_to_index(move, flip=flip)
                    if idx < 4096:
                        mask[idx] = 0.0
                probs = torch.softmax(logits + mask, dim=0)
                legal_move_probs = []
                for move in legal_moves:
                    idx = self.move_to_index(move, flip=flip)
                    p   = float(probs[idx]) if idx < 4096 else 1.0 / len(legal_moves)
                    legal_move_probs.append((move, p))
                total = sum(p for _, p in legal_move_probs)
                if total > 0 and abs(total - 1.0) > 1e-6:
                    legal_move_probs = [(m, p / total) for m, p in legal_move_probs]
                results.append((legal_move_probs, value))
            return results

    def get_move_probabilities(self, board):
        res = self.evaluate_batch([board])
        return res[0] if res else ([], 0.0)

    # -------------------------
    # Tree reuse
    # -------------------------
    def _get_reusable_root(self, board):
        if self._mcts_root_cache is None:
            return None, False
        target_fen  = board.fen()
        cached_root = self._mcts_root_cache
        for _, child in cached_root.children.items():
            if child.board.fen() == target_fen:
                child.parent = None
                return child, True
        for _, child in cached_root.children.items():
            for _, grandchild in child.children.items():
                if grandchild.board.fen() == target_fen:
                    grandchild.parent = None
                    return grandchild, True
        return None, False

    # -------------------------
    # Batched MCTS
    # -------------------------
    def run_mcts_batched(self, root_board, simulations=None, add_dirichlet_noise=False,
                         game_position_counts=None, reuse_tree=True):
        if simulations is None:
            simulations = self.mcts_simulations
        if game_position_counts is None:
            game_position_counts = {}

        reused_root = None
        if reuse_tree:
            reused_root, _ = self._get_reusable_root(root_board)

        if reused_root is not None:
            root = reused_root
            root.board = root_board.copy()
            if not root.children:
                for move, prob in self.get_move_probabilities(root.board)[0]:
                    b = root.board.copy(); b.push(move)
                    root.children[move] = self.MCTSNode(b, parent=root, prior=prob)
        else:
            root = self.MCTSNode(root_board.copy())
            for move, prob in self.get_move_probabilities(root.board)[0]:
                b = root.board.copy(); b.push(move)
                root.children[move] = self.MCTSNode(b, parent=root, prior=prob)

        if add_dirichlet_noise and root.children:
            eps   = self.mcts_dirichlet_eps
            alpha = self.mcts_dirichlet_alpha
            moves = list(root.children.keys())
            noise = np.random.dirichlet([alpha] * len(moves))
            for i, m in enumerate(moves):
                root.children[m].prior = (1 - eps) * root.children[m].prior + eps * noise[i]

        num_batches = (simulations + self.mcts_batch_size - 1) // self.mcts_batch_size

        for batch_idx in range(num_batches):
            bsize        = min(self.mcts_batch_size, simulations - batch_idx * self.mcts_batch_size)
            search_paths = []
            leaf_nodes   = []

            for _ in range(bsize):
                node = root
                path = [node]
                while node.children:
                    parent_q    = node.q_value
                    total_visits = sum(c.visits + c.virtual_loss_count
                                      for c in node.children.values()) + 1
                    best_score  = -1e9
                    best_move   = None
                    for move, child in node.children.items():
                        q = (parent_q - self.mcts_fpu_reduction) if child.visits == 0 \
                            else child.value_sum / child.visits
                        u = (self.mcts_c_puct * child.prior
                             * math.sqrt(total_visits)
                             / (1 + child.visits + child.virtual_loss_count))
                        if q + u > best_score:
                            best_score = q + u
                            best_move  = move
                    if best_move is None:
                        break
                    node = node.children[best_move]
                    node.virtual_loss_count += 1
                    path.append(node)
                search_paths.append(path)
                leaf_nodes.append(node)

            boards_to_eval  = []
            terminal_values = []
            eval_map        = []

            for idx, node in enumerate(leaf_nodes):
                if node.board.is_game_over():
                    res = node.board.result()
                    raw = 1.0 if res == "1-0" else (-1.0 if res == "0-1" else self.draw_penalty)
                    if node.turn != root_board.turn:
                        raw = -raw
                    terminal_values.append(raw)
                else:
                    boards_to_eval.append(node.board)
                    terminal_values.append(None)
                    eval_map.append(idx)

            eval_results = self.evaluate_batch(boards_to_eval) if boards_to_eval else []
            leaf_values  = [None] * len(leaf_nodes)

            for ei, node_idx in enumerate(eval_map):
                node             = leaf_nodes[node_idx]
                mv_probs, lv     = eval_results[ei]
                fen_key          = node.board.board_fen()
                prior_visits     = game_position_counts.get(fen_key, 0)
                if prior_visits >= 1:
                    lv = float(np.clip(lv + self.repetition_penalty * prior_visits, -1.0, 1.0))
                if node.turn != root_board.turn:
                    lv = -lv
                leaf_values[node_idx] = lv
                for move, prob in mv_probs:
                    if move not in node.children:
                        b = node.board.copy(); b.push(move)
                        node.children[move] = self.MCTSNode(b, parent=node, prior=prob)

            for idx, tv in enumerate(terminal_values):
                if tv is not None:
                    leaf_values[idx] = tv

            for path, lv in zip(search_paths, leaf_values):
                if lv is None:
                    continue
                val = lv
                for n in reversed(path):
                    n.visits      += 1
                    n.value_sum   += val
                    if n.virtual_loss_count > 0:
                        n.virtual_loss_count -= 1
                    val = -val

        self._mcts_root_cache = root
        return root

    # -------------------------
    # Move selection
    # -------------------------
    def select_move(self, board, temperature=1.0, use_mcts=True,
                    add_dirichlet_noise=False, game_position_counts=None):
        if not use_mcts:
            move_probs, _ = self.get_move_probabilities(board)
            if not move_probs:
                return None
            if temperature == 0:
                return max(move_probs, key=lambda x: x[1])[0]
            moves, probs = zip(*move_probs)
            probs = np.array(probs, dtype=np.float64)
            probs = np.clip(probs, 1e-12, None) ** (1.0 / temperature)
            probs /= probs.sum()
            return np.random.choice(moves, p=probs)

        root = self.run_mcts_batched(board, simulations=self.mcts_simulations,
                                     add_dirichlet_noise=add_dirichlet_noise,
                                     game_position_counts=game_position_counts,
                                     reuse_tree=True)
        if not root.children:
            return None
        moves  = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)
        if temperature == 0 or temperature < 1e-8:
            return moves[int(np.argmax(visits))]
        probs = visits ** (1.0 / temperature)
        probs /= probs.sum()
        return np.random.choice(moves, p=probs)

    # -------------------------
    # Self-play
    # -------------------------
    def play_game(self, temperature=1.0, max_moves=300, temp_threshold=30):
        board           = chess.Board()
        game_data       = []
        move_count      = 0
        position_counts = {}

        self.reset_mcts_tree()

        while not board.is_game_over() and not self.stop_training_flag:
            fen_key     = board.board_fen()
            visit_count = position_counts.get(fen_key, 0)
            position_counts[fen_key] = visit_count + 1

            if visit_count >= 1:
                inline_penalty = self.repetition_penalty * visit_count
                self.training_stats['repetition_penalties_applied'] += 1
            else:
                inline_penalty = 0.0

            board_tensor = self.board_to_tensor(board).cpu()
            current_temp = temperature if move_count < temp_threshold else 0.0
            move = self.select_move(board, temperature=current_temp, use_mcts=True,
                                    add_dirichlet_noise=True,
                                    game_position_counts=position_counts)
            if move is None:
                break

            # Augment all moves except castling
            can_flip = not board.is_castling(move)
            flip     = (board.turn == chess.BLACK)
            move_idx = self.move_to_index(move, flip=flip)
            player   = board.turn

            with torch.no_grad():
                self.model.eval()
                t = board_tensor.to(self.device)
                if self.device.type == 'cuda':
                    t = t.to(memory_format=torch.channels_last)
                with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                    policy_logits, _ = self.model(t)
                lp = F.log_softmax(policy_logits, dim=1)
                move_log_prob = lp[0, move_idx].item() if move_idx < lp.shape[1] else -10.0

            game_data.append((board_tensor, move_idx, player, move_log_prob, can_flip, inline_penalty))
            board.push(move)
            move_count += 1
            if len(game_data) > max_moves:
                break

        result = board.result()
        if result == "1-0":
            reward = 1.0;  self.training_stats['white_wins'] += 1
        elif result == "0-1":
            reward = -1.0; self.training_stats['black_wins'] += 1
        else:
            reward = self.draw_penalty; self.training_stats['draws'] += 1

        self.training_stats['games_played'] += 1
        self.training_stats['total_moves']  += len(game_data)
        self.reset_mcts_tree()
        return game_data, reward

    # -------------------------
    # Replay buffer
    # -------------------------
    def add_game_to_buffer(self, game_data, reward):
        for board_tensor, move_idx, player, old_log_prob, can_flip, inline_penalty in game_data:
            if move_idx is None:
                continue
            base_value   = reward if player == chess.WHITE else -reward
            target_value = float(np.clip(base_value + inline_penalty, -1.0, 1.0))
            augmented    = self.augment_tensor_and_index(board_tensor, move_idx, can_flip)
            self.training_stats['positions_total'] += 1
            if can_flip:
                self.training_stats['positions_flipped'] += 1
            for aug_tensor, aug_move_idx in augmented:
                self.replay_buffer.append((aug_tensor, aug_move_idx, target_value,
                                           old_log_prob, self.data_counter))
            self.data_counter += 1

    def clean_old_data(self):
        if len(self.replay_buffer) < self.replay_capacity:
            return
        cutoff     = self.data_counter
        new_buffer = deque(maxlen=self.replay_capacity)
        for entry in self.replay_buffer:
            if cutoff - entry[4] < self.max_data_age:
                new_buffer.append(entry)
        self.replay_buffer = new_buffer

    def sample_batch(self):
        if not self.replay_buffer:
            raise ValueError("Replay buffer is empty")
        batch = random.sample(self.replay_buffer, min(self.batch_size, len(self.replay_buffer)))
        if len(batch) < self.batch_size:
            batch = [random.choice(self.replay_buffer) for _ in range(self.batch_size)]
        boards, move_idxs, target_values, old_log_probs, _ = zip(*batch)
        boards_t      = torch.cat(boards, dim=0).to(self.device)
        if self.device.type == 'cuda':
            boards_t = boards_t.to(memory_format=torch.channels_last)
        move_idxs_t   = torch.LongTensor(move_idxs).to(self.device)
        target_vals_t = torch.FloatTensor(target_values).to(self.device)
        old_lp_t      = torch.FloatTensor(old_log_probs).to(self.device)
        return boards_t, move_idxs_t, target_vals_t, old_lp_t

    # -------------------------
    # Training step
    # -------------------------
    def train_on_batch(self, boards_t, move_idxs_t, target_vals_t, old_lp_t):
        self.model.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            policy_logits, values = self.model(boards_t)
            log_probs  = F.log_softmax(policy_logits, dim=1)
            sel_lp     = log_probs.gather(1, move_idxs_t.unsqueeze(1)).squeeze(1)
            sel_lp     = torch.clamp(sel_lp, -10.0, 0.0)
            ratio      = torch.clamp(torch.exp(sel_lp - old_lp_t), 0.5, 2.0)
            advantages = (target_vals_t - values).detach()
            adv_std    = advantages.std(unbiased=False)
            adv_norm   = ((advantages - advantages.mean()) / (adv_std + 1e-8)
                         if adv_std.item() >= 1e-6 else advantages - advantages.mean())
            adv_norm   = torch.clamp(adv_norm, -5.0, 5.0)
            p_loss     = torch.clamp(-(ratio * sel_lp * adv_norm).mean(), -10.0, 10.0)
            v_loss     = F.mse_loss(values, target_vals_t)
            probs      = F.softmax(policy_logits, dim=1)
            entropy    = -(probs * log_probs).sum(dim=1).mean()
            ec         = self.entropy_coef * (1.0 + 0.1 / (
                         1.0 + self.training_stats['total_training_steps'] / 1000.0))
            loss       = p_loss + self.value_coef * v_loss - ec * entropy

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

        self.loss_history.append(p_loss.item())
        self.training_stats['total_training_steps'] += 1
        try:
            self.scheduler.step()
        except Exception:
            pass
        return p_loss.item(), v_loss.item()

    def train(self, num_games=10, temperature=1.0, temp_threshold=30, callback=None):
        self.stop_training_flag = False
        for game_num in range(num_games):
            if self.stop_training_flag:
                break
            t0       = time.time()
            gd, rew  = self.play_game(temperature, temp_threshold=temp_threshold)
            game_t   = time.time() - t0

            self.add_game_to_buffer(gd, rew)
            if game_num % 10 == 0:
                self.clean_old_data()

            avg_p, avg_v, steps = 0.0, 0.0, 0
            t1 = time.time()
            if len(self.replay_buffer) >= max(self.min_buffer_size, self.batch_size):
                for _ in range(self.train_steps_per_game):
                    try:
                        b, m, v, lp = self.sample_batch()
                    except ValueError:
                        break
                    pl, vl = self.train_on_batch(b, m, v, lp)
                    if pl == pl:  # skip NaN
                        avg_p += pl; avg_v += vl; steps += 1
                if steps:
                    avg_p /= steps; avg_v /= steps
            train_t = time.time() - t1

            if callback:
                callback(game_num + 1, num_games, avg_p, avg_v, rew, game_t, train_t)
            if (game_num + 1) % 10 == 0:
                self.save_model()

    def stop_training(self):
        self.stop_training_flag = True

    # -------------------------
    # Save / Load
    # -------------------------
    def save_model(self):
        path = os.path.join(self.save_dir, "model_latest.pth")
        try:
            save_dict = {
                'model_state_dict':     {k: v.cpu() for k, v in self.model.state_dict().items()},
                'training_stats':       self.training_stats,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'net_channels':         self.model.input_conv.out_channels,
                'net_blocks':           len(self.model.tower),
            }
            if self.scaler:
                save_dict['scaler_state_dict'] = self.scaler.state_dict()
            torch.save(save_dict, path)
            if self.training_stats['games_played'] % 50 == 0:
                ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(save_dict, os.path.join(self.save_dir, f"model_{ts}.pth"))
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        path = os.path.join(self.save_dir, "model_latest.pth")
        if not os.path.exists(path):
            print("No saved model found. Starting fresh.")
            return
        try:
            ckpt = torch.load(path, map_location='cpu')
            sc   = ckpt.get('net_channels', None)
            sb   = ckpt.get('net_blocks',   None)
            cc   = self.model.input_conv.out_channels
            cb   = len(self.model.tower)

            if sc != cc or sb != cb:
                print(f"⚠️  Architecture mismatch (saved {sb}×{sc}, current {cb}×{cc}).")
                print("   Weights NOT loaded — starting fresh network.")
            else:
                self.model.load_state_dict(ckpt['model_state_dict'])
                self.model.to(self.device)
                if self.device.type == 'cuda':
                    self.model = self.model.to(memory_format=torch.channels_last)
                if 'optimizer_state_dict' in ckpt:
                    self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
                if 'scheduler_state_dict' in ckpt:
                    self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
                if self.scaler and 'scaler_state_dict' in ckpt:
                    self.scaler.load_state_dict(ckpt['scaler_state_dict'])
                print(f"Model loaded from {path}")

            self.training_stats = ckpt.get('training_stats', self.training_stats)
            for key in ['positions_flipped', 'positions_total',
                        'total_training_steps', 'repetition_penalties_applied']:
                self.training_stats.setdefault(key, 0)
            print(f"Stats: {self.training_stats}")
        except Exception as e:
            print(f"Error loading model: {e}")


# -------------------------
# GUI
# -------------------------
class ChessGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess AI — ResNet + Improved Augmentation")
        self.window.geometry("1000x800")

        self.ai                       = ChessAI()
        self.board                    = chess.Board()
        self.selected_square          = None
        self.legal_moves_for_selected = []
        self.human_color              = None
        self.is_training              = False
        self.training_thread          = None
        self.square_size              = 60
        self.move_history             = []
        self.ai_thinking              = False
        self.flip_board               = False
        self.flip_var                 = tk.BooleanVar(value=False)
        self.message_queue            = queue.Queue()
        self.ai_vs_ai_running         = False
        self.ai_vs_ai_paused          = False

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
        ttk.Button(history_frame, text="Copy Moves", command=self.copy_moves, width=15).grid(row=1, column=0, pady=2)

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

        self.stop_button = ttk.Button(button_frame, text="Stop Training",
                                      command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.progress_var = tk.StringVar(value="No training in progress")
        ttk.Label(train_frame, textvariable=self.progress_var,
                  wraplength=250).grid(row=4, column=0, columnspan=3, pady=5)

        play_frame = ttk.LabelFrame(right_frame, text="Play Controls", padding="10")
        play_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(play_frame, text="Play as White",
                   command=lambda: self.start_game(chess.WHITE), width=20).grid(row=0, column=0, pady=5)
        ttk.Button(play_frame, text="Play as Black",
                   command=lambda: self.start_game(chess.BLACK), width=20).grid(row=1, column=0, pady=5)
        ttk.Button(play_frame, text="AI vs AI Demo",
                   command=self.watch_ai_game, width=20).grid(row=2, column=0, pady=5)
        self.pause_button = ttk.Button(play_frame, text="Pause",
                                       command=self.toggle_pause_ai_game, width=20, state=tk.DISABLED)
        self.pause_button.grid(row=3, column=0, pady=5)
        ttk.Button(play_frame, text="New Game",
                   command=self.reset_game, width=20).grid(row=4, column=0, pady=5)
        ttk.Checkbutton(play_frame, text="Flip board",
                        variable=self.flip_var, command=self.on_flip_toggle).grid(row=5, column=0, pady=5)

        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)

        self.stats_text = tk.Text(stats_frame, height=16, width=35, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0)
        ttk.Button(stats_frame, text="Copy Stats",
                   command=self.copy_stats, width=15).grid(row=1, column=0, pady=2)

        amp_status = "AMP: ON" if self.ai.use_amp else "AMP: OFF"
        net_info   = f"Net: {len(self.ai.model.tower)} blocks × {self.ai.model.input_conv.out_channels}ch"
        ttk.Label(stats_frame, text=f"Device: {self.ai.device}",
                  foreground="blue").grid(row=2, column=0, pady=2)
        ttk.Label(stats_frame, text=amp_status,
                  foreground="green").grid(row=3, column=0, pady=2)
        ttk.Label(stats_frame, text=net_info, foreground="darkgreen",
                  font=('Arial', 9, 'bold')).grid(row=4, column=0, pady=2)
        ttk.Label(stats_frame, text="✅ ResNet + BN Architecture", foreground="purple",
                  font=('Arial', 9, 'bold')).grid(row=5, column=0, pady=2)
        ttk.Label(stats_frame, text="✅ MCTS: Tree Reuse + FPU", foreground="purple",
                  font=('Arial', 9, 'bold')).grid(row=6, column=0, pady=2)
        ttk.Label(stats_frame, text="✅ Aug: ~99% positions flipped", foreground="darkorange",
                  font=('Arial', 9, 'bold')).grid(row=7, column=0, pady=2)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var,
                  relief=tk.SUNKEN, anchor=tk.W).grid(row=1, column=0, columnspan=2,
                                                       sticky=(tk.W, tk.E), pady=5)

        self.update_board_display()
        self.update_stats_display()

    def on_flip_toggle(self):
        self.flip_board = bool(self.flip_var.get())
        self.update_board_display()

    def board_to_image(self):
        board_size = self.square_size * 8
        image = Image.new('RGB', (board_size + 40, board_size + 40), 'white')
        draw  = ImageDraw.Draw(image)

        light_sq   = (240, 217, 181)
        dark_sq    = (181, 136, 99)
        sel_color  = (255, 255, 100)
        legal_color= (144, 238, 144)
        offset     = 20

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

        legal_targets = {m.to_square for m in self.legal_moves_for_selected}

        for rank in range(8):
            for file in range(8):
                x1 = (file if not self.flip_board else 7 - file) * self.square_size + offset
                y1 = ((7 - rank) if not self.flip_board else rank) * self.square_size + offset
                sq = chess.square(file, rank)
                if sq == self.selected_square:
                    color = sel_color
                elif sq in legal_targets:
                    color = legal_color
                elif (rank + file) % 2 == 0:
                    color = light_sq
                else:
                    color = dark_sq
                draw.rectangle([x1, y1, x1 + self.square_size, y1 + self.square_size],
                                fill=color, outline='gray')

        files = ['a','b','c','d','e','f','g','h']
        ranks = ['1','2','3','4','5','6','7','8']
        if self.flip_board:
            files = list(reversed(files))
            ranks = list(reversed(ranks))

        for i, fc in enumerate(files):
            x = i * self.square_size + self.square_size // 2 + offset
            draw.text((x, 5), fc, fill='black', font=coord_font, anchor='mm')
            draw.text((x, board_size + offset + 15), fc, fill='black', font=coord_font, anchor='mm')

        for i, rc in enumerate(ranks):
            y = ((7 - i) if not self.flip_board else i) * self.square_size + self.square_size // 2 + offset
            draw.text((5, y), rc, fill='black', font=coord_font, anchor='mm')
            draw.text((board_size + offset + 15, y), rc, fill='black', font=coord_font, anchor='mm')

        piece_symbols = {
            'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
            'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'
        }

        for rank in range(8):
            for file in range(8):
                piece = self.board.piece_at(chess.square(file, rank))
                if piece:
                    pc = piece_symbols.get(piece.symbol(), piece.symbol())
                    x  = (file if not self.flip_board else 7 - file) * self.square_size + self.square_size // 2 + offset
                    y  = ((7 - rank) if not self.flip_board else rank) * self.square_size + self.square_size // 2 + offset
                    fg = 'white' if piece.color == chess.WHITE else 'black'
                    bg = 'black' if piece.color == chess.WHITE else 'white'
                    for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                        draw.text((x+dx, y+dy), pc, fill=bg, font=piece_font, anchor='mm')
                    draw.text((x, y), pc, fill=fg, font=piece_font, anchor='mm')

        return ImageTk.PhotoImage(image)

    def update_board_display(self):
        try:
            photo = self.board_to_image()
            self.board_label.config(image=photo)
            self.board_label.image = photo
        except Exception as e:
            self.board_label.config(text=str(self.board))

    def update_stats_display(self):
        s   = self.ai.training_stats
        gp  = s['games_played']
        ww  = s['white_wins'] / gp * 100 if gp else 0
        bw  = s['black_wins'] / gp * 100 if gp else 0
        dr  = s['draws']      / gp * 100 if gp else 0
        pt  = s['positions_total']
        fr  = s['positions_flipped'] / pt * 100 if pt else 0
        lr  = self.ai.optimizer.param_groups[0]['lr']
        rep = s.get('repetition_penalties_applied', 0)
        pm  = sum(p.numel() for p in self.ai.model.parameters()) / 1e6

        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, f"""Games: {gp}
Moves: {s['total_moves']}
Steps: {s.get('total_training_steps', 0)}

White: {s['white_wins']} ({ww:.1f}%)
Black: {s['black_wins']} ({bw:.1f}%)
Draws: {s['draws']} ({dr:.1f}%)

Buffer: {len(self.ai.replay_buffer)}
Augmentation: {fr:.1f}% flipped
Rep.penalties: {rep}
LR: {lr:.2e}
Draw penalty: {self.ai.draw_penalty}
Rep penalty: {self.ai.repetition_penalty}
FPU reduction: {self.ai.mcts_fpu_reduction}
Params: {pm:.2f}M
Model: {self.ai.save_dir}""")

    def update_move_history(self):
        self.history_text.delete(1.0, tk.END)
        if not self.move_history:
            self.history_text.insert(1.0, "No moves yet")
            return
        text = ""
        for i, mv in enumerate(self.move_history):
            text += f"{i//2+1}. {mv} " if i % 2 == 0 else f"{mv}\n"
        self.history_text.insert(1.0, text)
        self.history_text.see(tk.END)

    def on_board_click(self, event):
        if (self.human_color is None or self.board.turn != self.human_color
                or self.board.is_game_over() or self.ai_thinking):
            return
        offset = 20
        x, y   = event.x - offset, event.y - offset
        if not (0 <= x < self.square_size * 8 and 0 <= y < self.square_size * 8):
            return
        col  = int(min(7, max(0, x // self.square_size)))
        row  = int(min(7, max(0, y // self.square_size)))
        file = col       if not self.flip_board else 7 - col
        rank = 7 - row   if not self.flip_board else row
        sq   = chess.square(file, rank)

        if self.selected_square is None:
            piece = self.board.piece_at(sq)
            if piece and piece.color == self.human_color:
                self.selected_square          = sq
                self.legal_moves_for_selected = [m for m in self.board.legal_moves
                                                  if m.from_square == sq]
                self.status_var.set(f"Selected {chess.SQUARE_NAMES[sq]}")
                self.update_board_display()
        else:
            piece = self.board.piece_at(self.selected_square)
            promo = None
            if piece and piece.piece_type == chess.PAWN and (rank == 0 or rank == 7):
                promo = self.ask_promotion_piece() or chess.QUEEN
            move = chess.Move(self.selected_square, sq, promotion=promo)
            if move in self.board.legal_moves:
                self.make_move(move)
                self.selected_square          = None
                self.legal_moves_for_selected = []
                self.update_board_display()
                if self.board.is_game_over():
                    self.game_over()
                else:
                    self.window.after(300, self.ai_move)
            else:
                self.selected_square          = None
                self.legal_moves_for_selected = []
                self.status_var.set("Illegal move")
                self.update_board_display()

    def ask_promotion_piece(self):
        dlg    = tk.Toplevel(self.window)
        dlg.title("Promotion")
        dlg.transient(self.window)
        dlg.grab_set()
        choice = {'piece': None}
        ttk.Label(dlg, text="Promote to:", padding=10).grid(row=0, column=0, columnspan=4)
        for col, (sym, pc) in enumerate([('♕', chess.QUEEN), ('♖', chess.ROOK),
                                          ('♗', chess.BISHOP), ('♘', chess.KNIGHT)]):
            ttk.Button(dlg, text=sym, width=8,
                       command=lambda p=pc: [choice.update({'piece': p}), dlg.destroy()]
                       ).grid(row=1, column=col, padx=5, pady=5)
        dlg.wait_window()
        return choice['piece']

    def make_move(self, move):
        san = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san)
        self.update_move_history()

    def ai_move(self):
        if self.board.is_game_over():
            self.game_over(); return
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        self.window.update()
        try:
            self.ai.reset_mcts_tree()
            move = self.ai.select_move(self.board, temperature=0.0,
                                       use_mcts=True, add_dirichlet_noise=False)
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
        self.board                    = chess.Board()
        self.human_color              = color
        self.selected_square          = None
        self.legal_moves_for_selected = []
        self.move_history             = []
        self.ai_thinking              = False
        self.flip_board               = (color == chess.BLACK)
        self.flip_var.set(self.flip_board)
        self.ai.reset_mcts_tree()
        self.update_board_display()
        self.update_move_history()
        self.status_var.set(f"You are {'White' if color == chess.WHITE else 'Black'}")
        if color == chess.BLACK:
            self.window.after(500, self.ai_move)

    def watch_ai_game(self):
        self.board                    = chess.Board()
        self.human_color              = None
        self.selected_square          = None
        self.legal_moves_for_selected = []
        self.move_history             = []
        self.flip_board               = False
        self.flip_var.set(False)
        self.ai_vs_ai_running         = True
        self.ai_vs_ai_paused          = False
        self.ai.reset_mcts_tree()
        self.pause_button.config(text="Pause", state=tk.NORMAL)
        self.update_board_display()
        self.update_move_history()
        self.play_ai_vs_ai()

    def play_ai_vs_ai(self):
        if not self.ai_vs_ai_running:
            return
        if self.ai_vs_ai_paused:
            self.window.after(200, self.play_ai_vs_ai); return
        if not self.board.is_game_over():
            try:
                move = self.ai.select_move(self.board, temperature=0.1, use_mcts=True)
                if move:
                    self.make_move(move)
                    self.update_board_display()
                    self.window.after(800, self.play_ai_vs_ai)
            except Exception as e:
                self.status_var.set(f"Error: {e}")
                self.ai_vs_ai_running = False
                self.pause_button.config(state=tk.DISABLED)
        else:
            self.ai_vs_ai_running = False
            self.pause_button.config(text="Pause", state=tk.DISABLED)
            self.game_over()

    def reset_game(self):
        self.ai_vs_ai_running         = False
        self.ai_vs_ai_paused          = False
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.board                    = chess.Board()
        self.human_color              = None
        self.selected_square          = None
        self.legal_moves_for_selected = []
        self.move_history             = []
        self.flip_board               = False
        self.flip_var.set(False)
        self.ai.reset_mcts_tree()
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Ready")

    def game_over(self):
        outcome = self.board.outcome()
        msg = ("Game ended" if not outcome
               else "White wins" if outcome.winner == chess.WHITE
               else "Black wins" if outcome.winner == chess.BLACK
               else "Draw")
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
        result = ("Draw" if abs(reward - self.ai.draw_penalty) < 1e-6
                  else "Win(W)" if reward > 0 else "Win(B)")
        self.message_queue.put({'type': 'training_update',
                                'text': (f"Game {game_num}/{total}\n{result}\n"
                                         f"P: {p_loss:.3f} V: {v_loss:.3f}\n"
                                         f"Game: {game_t:.1f}s Train: {train_t:.1f}s")})

    def train_worker(self, num, temp, temp_threshold):
        try:
            self.ai.train(num_games=num, temperature=temp,
                          temp_threshold=temp_threshold, callback=self.training_callback)
            self.message_queue.put({'type': 'training_complete', 'text': f"Trained {num} games"})
        except Exception as e:
            self.message_queue.put({'type': 'training_error', 'text': str(e)})

    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Already training"); return
        try:
            num           = int(self.num_games_var.get())
            temp          = float(self.temperature_var.get())
            temp_threshold= int(self.temp_threshold_var.get())
            if num <= 0 or temp <= 0 or temp_threshold < 0:
                raise ValueError
        except:
            messagebox.showerror("Error", "Invalid input"); return
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(f"Starting... (Temp={temp}, Switch@move {temp_threshold})")
        self.training_thread = threading.Thread(
            target=self.train_worker, args=(num, temp, temp_threshold), daemon=True)
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

    def toggle_pause_ai_game(self):
        if not self.ai_vs_ai_running:
            return
        self.ai_vs_ai_paused = not self.ai_vs_ai_paused
        self.pause_button.config(text="Resume" if self.ai_vs_ai_paused else "Pause")
        self.status_var.set("AI vs AI paused" if self.ai_vs_ai_paused else "AI vs AI running...")

    def copy_stats(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.stats_text.get(1.0, tk.END).strip())
        self.status_var.set("Stats copied to clipboard")

    def copy_moves(self):
        self.window.clipboard_clear()
        self.window.clipboard_append(self.history_text.get(1.0, tk.END).strip())
        self.status_var.set("Move history copied to clipboard")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()