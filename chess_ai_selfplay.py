import chess
import chess.engine
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import random
import pickle
import os
import math
import shutil
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
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        return F.relu(x + residual)


class ChessNet(nn.Module):
    def __init__(self, channels=128, num_res_blocks=6):
        super(ChessNet, self).__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(19, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )
        self.res_blocks = nn.Sequential(
            *[ResidualBlock(channels) for _ in range(num_res_blocks)]
        )

        # AlphaZero-style policy head: keep spatial features until the final move map.
        self.policy_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 4096),  # 64*64 possible from-to moves
        )

        # AlphaZero-style value head.
        self.value_head = nn.Sequential(
            nn.Conv2d(channels, 32, kernel_size=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 1),
            nn.Tanh(),
        )

        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                if hasattr(m, "weight") and m.weight is not None:
                    nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if hasattr(m, "bias") and m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        x = self.stem(x)
        x = self.res_blocks(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
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
            self.legal_moves = None
            self.legal_indices = None

        @property
        def q_value(self):
            total_visits = self.visits + self.virtual_loss_count
            if total_visits == 0:
                return 0.0
            return self.value_sum / total_visits

    def __init__(self, save_dir="chess_ai_models",
                 replay_capacity=30000,
                 batch_size=128,
                 train_steps_per_game=64,
                 entropy_coef=0.01,
                 value_coef=1.5,
                 clip_grad=1.0,
                 min_buffer_size=200,
                 lr=1e-4,
                 weight_decay=1e-4,
                 max_data_age=2000,
                 draw_penalty=0.0,
                 repetition_penalty=-0.05,
                 repetition_draw_penalty=0.0,
                 mcts_simulations=1024,
                 mcts_batch_size=64,
                 mcts_c_puct=1.4,
                 mcts_dirichlet_eps=0.25,
                 mcts_dirichlet_alpha=0.3,
                 stockfish_path="c:\\Users\\libby\\Downloads\\stockfish-windows-x86-64-avx2\\stockfish\\stockfish-windows-x86-64-avx2.exe",
                 stockfish_time_limit=1,
                 stockfish_depth=24,
                 stockfish_top_moves=5,
                 stockfish_teacher_start=0.85,
                 stockfish_teacher_end=0.20,
                 stockfish_teacher_decay_games=10000,
                 prioritized_replay_alpha=0.6,
                 priority_epsilon=1e-3,
                 human_policy_weight=0.25,
                 use_amp=True,
                 resignation_threshold=-0.9,
                 resignation_consecutive_plies=5,
                 opening_random_plies=2,
                 evaluation_games=4,
                 evaluation_interval=50,
                 parallel_games=8):

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
            'total_game_time': 0.0,
            'total_train_time': 0.0,
            'last_game_time': 0.0,
            'last_train_time': 0.0,
            'last_game_moves': 0,
            'last_draw_reason': '',
            'rule_draws': 0,
            'max_move_draws': 0,
            'human_games': 0,
            'human_positions': 0,
            'human_train_steps': 0,
            'stockfish_games': 0,
            'stockfish_positions': 0,
            'stockfish_unavailable_games': 0,
            'resigned_games': 0,
            'evaluation_games': 0,
            'evaluation_wins': 0,
            'evaluation_losses': 0,
            'evaluation_draws': 0,
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
        self.repetition_draw_penalty = repetition_draw_penalty
        
        self.mcts_simulations = mcts_simulations
        self.mcts_batch_size = mcts_batch_size
        self.mcts_c_puct = mcts_c_puct
        self.mcts_dirichlet_eps = mcts_dirichlet_eps
        self.mcts_dirichlet_alpha = mcts_dirichlet_alpha
        self.stockfish_path = (
            stockfish_path
            if stockfish_path is not None
            else os.environ.get("STOCKFISH_PATH") or shutil.which("stockfish")
        )
        self.stockfish_time_limit = stockfish_time_limit
        self.stockfish_depth = stockfish_depth
        self.stockfish_top_moves = stockfish_top_moves
        self.stockfish_teacher_start = stockfish_teacher_start
        self.stockfish_teacher_end = stockfish_teacher_end
        self.stockfish_teacher_decay_games = stockfish_teacher_decay_games
        self.stockfish_engine = None
        self.stockfish_disabled_reason = ""
        self.last_evaluation_replays = []
        self.evaluation_replay_version = 0
        self.prioritized_replay_alpha = prioritized_replay_alpha
        self.priority_epsilon = priority_epsilon
        self.human_policy_weight = human_policy_weight
        self.resignation_threshold = float(resignation_threshold)
        self.resignation_consecutive_plies = max(1, int(resignation_consecutive_plies))
        self.opening_random_plies = max(0, int(opening_random_plies))
        self.evaluation_games = max(0, int(evaluation_games))
        self.evaluation_interval = max(1, int(evaluation_interval))
        self.parallel_games = max(1, int(parallel_games))
        
        self.data_counter = 0
        self.loss_history = deque(maxlen=100)
        
        os.makedirs(save_dir, exist_ok=True)
        self.load_model()

    def new_training_stats(self):
        return {
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'total_training_steps': 0,
            'positions_flipped': 0,
            'positions_total': 0,
            'repetition_penalties_applied': 0,
            'total_game_time': 0.0,
            'total_train_time': 0.0,
            'last_game_time': 0.0,
            'last_train_time': 0.0,
            'last_game_moves': 0,
            'last_draw_reason': '',
            'rule_draws': 0,
            'max_move_draws': 0,
            'human_games': 0,
            'human_positions': 0,
            'human_train_steps': 0,
            'stockfish_games': 0,
            'stockfish_positions': 0,
            'stockfish_unavailable_games': 0,
            'resigned_games': 0,
            'evaluation_games': 0,
            'evaluation_wins': 0,
            'evaluation_losses': 0,
            'evaluation_draws': 0,
        }

    def reset_learning_state(self, archive_checkpoint=True):
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        if archive_checkpoint and os.path.exists(model_path):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = os.path.join(self.save_dir, f"model_reset_backup_{timestamp}.pth")
            os.replace(model_path, backup_path)

        self.model = ChessNet().to(self.device)
        if self.device.type == 'cuda':
            self.model = self.model.to(memory_format=torch.channels_last)
        self.optimizer = optim.Adam(self.model.parameters(), lr=1e-4, weight_decay=1e-4)

        warmup_steps = 1000
        def lr_lambda(step):
            if step < warmup_steps:
                return float(step + 1) / float(max(1, warmup_steps))
            progress = min(1.0, float(step - warmup_steps) / float(max(1, self.scheduler_total_steps - warmup_steps)))
            return 0.5 * (1.0 + math.cos(math.pi * progress))

        self.scheduler = optim.lr_scheduler.LambdaLR(self.optimizer, lr_lambda)
        self.scaler = torch.cuda.amp.GradScaler() if self.use_amp else None
        self.training_stats = self.new_training_stats()
        self.replay_buffer.clear()
        self.data_counter = 0
        self.loss_history.clear()
    
    # -------------------------
    # Board / move helpers
    # -------------------------
    def board_to_tensor(self, board):
        tensor = np.zeros((19, 8, 8), dtype=np.float32)
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

        tensor[12, :, :] = 1.0 if current_player == chess.WHITE else 0.0
        own_color = current_player
        opp_color = not current_player
        tensor[13, :, :] = 1.0 if board.has_kingside_castling_rights(own_color) else 0.0
        tensor[14, :, :] = 1.0 if board.has_queenside_castling_rights(own_color) else 0.0
        tensor[15, :, :] = 1.0 if board.has_kingside_castling_rights(opp_color) else 0.0
        tensor[16, :, :] = 1.0 if board.has_queenside_castling_rights(opp_color) else 0.0

        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            ep_rank = chess.square_rank(board.ep_square)
            if current_player == chess.BLACK:
                ep_rank = 7 - ep_rank
            tensor[17, ep_rank, ep_file] = 1.0

        tensor[18, :, :] = min(1.0, board.halfmove_clock / 100.0)
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

    def position_key(self, board):
        return " ".join(board.fen().split(" ")[:4])

    def layout_key(self, board):
        return board.board_fen()

    def new_repetition_tracker(self):
        return {'exact': {}, 'layout': {}}

    def repetition_count_for_board(self, board, position_counts):
        if not position_counts:
            return 0

        if 'exact' in position_counts and 'layout' in position_counts:
            exact_count = position_counts['exact'].get(self.position_key(board), 0)
            layout_count = position_counts['layout'].get(self.layout_key(board), 0)
            return max(exact_count, layout_count)

        return position_counts.get(self.position_key(board), 0)

    def record_position_visit(self, board, position_counts):
        if 'exact' in position_counts and 'layout' in position_counts:
            exact_key = self.position_key(board)
            layout_key = self.layout_key(board)
            position_counts['exact'][exact_key] = position_counts['exact'].get(exact_key, 0) + 1
            position_counts['layout'][layout_key] = position_counts['layout'].get(layout_key, 0) + 1
            return

        position_key = self.position_key(board)
        position_counts[position_key] = position_counts.get(position_key, 0) + 1

    def repetition_penalty_for_visits(self, visit_count):
        if visit_count <= 0:
            return 0.0
        return self.repetition_penalty * visit_count

    def move_repetition_counts(self, board, moves, position_counts):
        if not position_counts:
            return np.zeros(len(moves), dtype=np.int32)

        repeat_counts = []
        for move in moves:
            next_board = board.copy()
            next_board.push(move)
            repeat_counts.append(self.repetition_count_for_board(next_board, position_counts))
        return np.array(repeat_counts, dtype=np.int32)

    def avoid_repeated_position_probs(self, board, moves, probs, position_counts):
        if not position_counts or not moves:
            return probs

        repeat_counts = self.move_repetition_counts(board, moves, position_counts)
        if not repeat_counts.any():
            return probs

        probs = np.asarray(probs, dtype=np.float64).copy()
        fresh_mask = repeat_counts == 0
        if fresh_mask.any():
            probs[~fresh_mask] = 0.0
            if float(probs.sum(dtype=np.float64)) <= 0.0:
                probs[fresh_mask] = 1.0
        else:
            probs /= (1.0 + repeat_counts.astype(np.float64))
        return probs

    def move_into_repetition_penalty(self, board, move, position_counts):
        if not position_counts or move is None:
            return 0.0

        next_board = board.copy()
        next_board.push(move)
        visit_count = self.repetition_count_for_board(next_board, position_counts)
        return self.repetition_penalty_for_visits(visit_count)

    def is_repetition_draw(self, board):
        return board.is_fivefold_repetition() or board.can_claim_threefold_repetition()

    def is_terminal_for_training(self, board):
        return board.is_game_over() or board.can_claim_threefold_repetition()

    def draw_value_for_board(self, board):
        if self.is_repetition_draw(board):
            return self.repetition_draw_penalty
        return self.draw_penalty

    def is_draw_reward(self, reward):
        return (
            abs(reward - self.draw_penalty) < 1e-6 or
            abs(reward - self.repetition_draw_penalty) < 1e-6
        )

    def terminal_value_for_side_to_move(self, board):
        outcome = board.outcome(claim_draw=True)
        if outcome and outcome.winner == chess.WHITE:
            return 1.0 if board.turn == chess.WHITE else -1.0
        if outcome and outcome.winner == chess.BLACK:
            return 1.0 if board.turn == chess.BLACK else -1.0
        return self.draw_value_for_board(board)

    def is_draw_result(self, board):
        outcome = board.outcome(claim_draw=True)
        return outcome is not None and outcome.winner is None

    # -------------------------
    # Stockfish teacher
    # -------------------------
    def stockfish_available(self):
        return bool(self.stockfish_path) and not self.stockfish_disabled_reason

    def stockfish_limit(self):
        kwargs = {}
        if self.stockfish_time_limit and self.stockfish_time_limit > 0:
            kwargs['time'] = self.stockfish_time_limit
        if self.stockfish_depth and self.stockfish_depth > 0:
            kwargs['depth'] = self.stockfish_depth
        return chess.engine.Limit(**kwargs)

    def get_stockfish_engine(self):
        if not self.stockfish_path:
            self.stockfish_disabled_reason = "Stockfish path is empty"
            return None
        if self.stockfish_disabled_reason:
            return None
        if self.stockfish_engine is not None:
            return self.stockfish_engine
        try:
            self.stockfish_engine = chess.engine.SimpleEngine.popen_uci(self.stockfish_path)
            return self.stockfish_engine
        except Exception as e:
            self.stockfish_disabled_reason = str(e)
            self.stockfish_engine = None
            return None

    def close_stockfish_engine(self):
        if self.stockfish_engine is None:
            return
        try:
            self.stockfish_engine.quit()
        except Exception:
            pass
        self.stockfish_engine = None

    def stockfish_teacher_rate(self):
        start = float(np.clip(self.stockfish_teacher_start, 0.0, 1.0))
        end = float(np.clip(self.stockfish_teacher_end, 0.0, 1.0))
        decay_games = max(1, int(self.stockfish_teacher_decay_games))
        progress = min(1.0, self.training_stats.get('games_played', 0) / decay_games)
        return start + (end - start) * progress

    def should_use_stockfish_teacher(self):
        if self.stockfish_teacher_rate() <= 0.0:
            return False
        if not self.stockfish_path or self.stockfish_disabled_reason:
            return False
        return random.random() < self.stockfish_teacher_rate()

    def stockfish_score_to_value(self, score, board):
        if score is None:
            return 0.0
        cp = score.pov(board.turn).score(mate_score=100000)
        if cp is None:
            return 0.0
        return float(np.tanh(np.clip(cp, -2000, 2000) / 600.0))

    def stockfish_policy_value(self, board, top_moves=None):
        engine = self.get_stockfish_engine()
        if engine is None:
            return None

        legal_count = board.legal_moves.count()
        if legal_count <= 0:
            return None
        multipv = max(1, min(int(top_moves or self.stockfish_top_moves), legal_count))

        try:
            infos = engine.analyse(board, self.stockfish_limit(), multipv=multipv)
        except Exception as e:
            self.stockfish_disabled_reason = str(e)
            self.close_stockfish_engine()
            return None

        if isinstance(infos, dict):
            infos = [infos]

        move_scores = []
        best_score = None
        teacher_value = 0.0
        for info in infos:
            pv = info.get('pv') or []
            score = info.get('score')
            if not pv or score is None:
                continue
            move = pv[0]
            if move not in board.legal_moves:
                continue
            cp = score.pov(board.turn).score(mate_score=100000)
            if cp is None:
                continue
            cp = float(np.clip(cp, -2000, 2000))
            if best_score is None or cp > best_score:
                best_score = cp
                teacher_value = self.stockfish_score_to_value(score, board)
            move_scores.append((move, cp))

        if not move_scores:
            return None

        scores = np.array([score for _, score in move_scores], dtype=np.float64)
        weights = np.exp((scores - scores.max()) / 60.0)
        probs = self.normalize_probabilities(weights)
        return [(move, float(prob)) for (move, _), prob in zip(move_scores, probs)], teacher_value

    def move_probs_to_policy_target(self, board, move_probs):
        flip = (board.turn == chess.BLACK)
        return tuple(
            (self.move_to_index(move, flip=flip), float(prob))
            for move, prob in move_probs
            if prob > 0.0
        )

    def verified_human_policy_target(self, board, move, fallback_policy):
        stockfish_result = self.stockfish_policy_value(board)
        if stockfish_result is None:
            return fallback_policy, self.human_policy_weight

        stockfish_move_probs, _ = stockfish_result
        human_idx = self.move_to_index(move, flip=(board.turn == chess.BLACK))
        stockfish_target = self.move_probs_to_policy_target(board, stockfish_move_probs)
        stockfish_indices = {idx for idx, _ in stockfish_target}

        if human_idx in stockfish_indices:
            blended = {idx: prob * 0.75 for idx, prob in stockfish_target}
            blended[human_idx] = blended.get(human_idx, 0.0) + 0.25
            return tuple((idx, prob) for idx, prob in blended.items()), 0.75

        return stockfish_target, self.human_policy_weight

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
            move_idx_flipped = self.flip_move_index_horizontal(move_idx)
            augmented.append((flipped_tensor, move_idx_flipped))
        return augmented

    def flip_move_index_horizontal(self, move_idx):
        from_sq = move_idx // 64
        to_sq = move_idx % 64
        from_file, from_rank = from_sq % 8, from_sq // 8
        to_file, to_rank = to_sq % 8, to_sq // 8
        from_sq_flipped = (7 - from_file) + from_rank * 8
        to_sq_flipped = (7 - to_file) + to_rank * 8
        return from_sq_flipped * 64 + to_sq_flipped

    def augment_tensor_and_policy(self, board_tensor, policy_target, can_flip=False):
        augmented = [(board_tensor.clone(), policy_target)]
        if can_flip:
            flipped_tensor = torch.flip(board_tensor, [3])
            flipped_policy = tuple(
                (self.flip_move_index_horizontal(move_idx), prob)
                for move_idx, prob in policy_target
            )
            augmented.append((flipped_tensor, flipped_policy))
        return augmented

    def legal_policy_indices(self, board):
        flip = (board.turn == chess.BLACK)
        return tuple(self.move_to_index(move, flip=flip) for move in board.legal_moves)

    def invalid_policy_logit(self, logits):
        # -1e9 overflows under CUDA AMP fp16; this value is safely representable.
        if logits.dtype == torch.float16:
            return -1e4
        return torch.finfo(logits.dtype).min

    def normalize_probabilities(self, probs):
        try:
            probs = np.asarray(probs, dtype=np.float64).reshape(-1)
        except Exception:
            return np.array([], dtype=np.float64)
        if probs.size == 0:
            return probs

        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(probs.shape, 1.0 / probs.size, dtype=np.float64)

        probs = probs / total
        probs = np.nan_to_num(probs, nan=0.0, posinf=0.0, neginf=0.0)
        probs = np.clip(probs, 0.0, None)
        total = float(probs.sum(dtype=np.float64))
        if not np.isfinite(total) or total <= 0.0:
            return np.full(probs.shape, 1.0 / probs.size, dtype=np.float64)

        probs = probs / total
        # Compensate for tiny floating point drift so numpy accepts p exactly.
        probs[-1] += 1.0 - float(probs.sum(dtype=np.float64))
        if not np.all(np.isfinite(probs)) or np.any(probs < 0.0):
            return np.full(probs.shape, 1.0 / probs.size, dtype=np.float64)
        return probs

    def safe_choice_index(self, count, probs):
        if count <= 0:
            return None
        probs = self.normalize_probabilities(probs)
        if probs.size != count or not np.all(np.isfinite(probs)):
            probs = np.full(count, 1.0 / count, dtype=np.float64)
        try:
            return int(np.random.choice(count, p=probs))
        except ValueError:
            return int(np.random.randint(count))

    def compact_training_targets(self, policy_target, legal_indices):
        policy_by_idx = {}
        for move_idx, prob in policy_target:
            if 0 <= move_idx < 4096:
                policy_by_idx[move_idx] = policy_by_idx.get(move_idx, 0.0) + float(prob)

        total_prob = sum(policy_by_idx.values())
        if total_prob > 0:
            policy_items = [
                (idx, prob / total_prob)
                for idx, prob in policy_by_idx.items()
                if prob > 0.0
            ]
        else:
            policy_items = []

        if policy_items:
            policy_indices, policy_probs = zip(*policy_items)
            policy_indices = torch.tensor(policy_indices, dtype=torch.long)
            policy_probs = torch.tensor(policy_probs, dtype=torch.float32)
        else:
            policy_indices = torch.empty(0, dtype=torch.long)
            policy_probs = torch.empty(0, dtype=torch.float32)

        legal_indices = sorted({idx for idx in legal_indices if 0 <= idx < 4096})
        legal_indices = torch.tensor(legal_indices, dtype=torch.long)
        return (policy_indices, policy_probs), legal_indices

    def flip_index_tuple_horizontal(self, move_indices):
        return tuple(self.flip_move_index_horizontal(move_idx) for move_idx in move_indices)

    def augment_training_entry(self, board_tensor, policy_target, legal_indices, can_flip=False):
        augmented = [(board_tensor.clone(), policy_target, legal_indices)]
        if can_flip:
            flipped_tensor = torch.flip(board_tensor, [3])
            flipped_policy = tuple(
                (self.flip_move_index_horizontal(move_idx), prob)
                for move_idx, prob in policy_target
            )
            flipped_legal = self.flip_index_tuple_horizontal(legal_indices)
            augmented.append((flipped_tensor, flipped_policy, flipped_legal))
        return augmented

    # -------------------------
    # Batched network inference
    # -------------------------
    def evaluate_batch(self, board_list, nodes=None, model=None):
        if len(board_list) == 0:
            return []
        if nodes is not None and len(nodes) != len(board_list):
            raise ValueError("nodes must match board_list length")
        
        model = self.model if model is None else model
        model.eval()
        with torch.no_grad():
            board_tensors = [self.board_to_tensor(board) for board in board_list]
            batch_tensor = torch.cat(board_tensors, dim=0).to(self.device)
            if self.device.type == 'cuda':
                batch_tensor = batch_tensor.to(memory_format=torch.channels_last)

            legal_moves_batch = []
            legal_indices_batch = []
            for i, board in enumerate(board_list):
                node = nodes[i] if nodes is not None else None
                if node is not None and node.legal_moves is not None:
                    legal_moves = node.legal_moves
                    legal_indices = node.legal_indices
                else:
                    flip = (board.turn == chess.BLACK)
                    legal_moves = list(board.legal_moves)
                    legal_indices = [
                        self.move_to_index(move, flip=flip)
                        for move in legal_moves
                    ]
                    if node is not None:
                        node.legal_moves = legal_moves
                        node.legal_indices = legal_indices
                legal_moves_batch.append(legal_moves)
                legal_indices_batch.append([
                    idx for idx in legal_indices
                    if 0 <= idx < 4096
                ])

            with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
                policy_logits, values = model(batch_tensor)

            policy_logits = torch.nan_to_num(
                policy_logits.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4
            )
            values = torch.nan_to_num(
                values.float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            ).cpu().numpy()
            legal_mask = torch.zeros(
                (len(board_list), 4096),
                dtype=torch.bool,
                device=self.device
            )
            row_parts = []
            col_parts = []
            for row, legal_indices in enumerate(legal_indices_batch):
                if legal_indices:
                    col_tensor = torch.tensor(
                        legal_indices,
                        dtype=torch.long,
                        device=self.device
                    )
                    row_parts.append(torch.full_like(col_tensor, row))
                    col_parts.append(col_tensor)
            if row_parts:
                rows = torch.cat(row_parts)
                cols = torch.cat(col_parts)
                legal_mask[rows, cols] = True

            masked_logits = policy_logits.masked_fill(
                ~legal_mask,
                self.invalid_policy_logit(policy_logits)
            )
            move_probs_tensor = F.softmax(masked_logits, dim=1)
            
            results = []
            for i, board in enumerate(board_list):
                value = float(values[i])
                legal_moves = legal_moves_batch[i]
                legal_indices = legal_indices_batch[i]
                legal_move_probs = []
                if legal_indices:
                    idx_tensor = torch.tensor(
                        legal_indices,
                        dtype=torch.long,
                        device=self.device
                    )
                    probs = move_probs_tensor[i, idx_tensor].float().cpu().numpy()
                    probs = self.normalize_probabilities(probs)
                else:
                    probs = np.array([], dtype=np.float64)
                for move, prob in zip(legal_moves, probs):
                    legal_move_probs.append((move, float(prob)))
                results.append((legal_move_probs, value))
            return results
    
    def get_move_probabilities(self, board, model=None):
        results = self.evaluate_batch([board], model=model)
        return results[0] if results else ([], 0.0)
    
    # -------------------------
    # Batched MCTS
    # -------------------------
    def run_mcts_batched(self, root_board, simulations=None, add_dirichlet_noise=False,
                         game_position_counts=None, model=None):
        """
        Run MCTS with batched neural network evaluation and CORRECT virtual loss.

        game_position_counts: dict of {position_key: visit_count} from the current game.
        Leaf nodes whose position has already been seen in the real game get their
        value penalized directly, so MCTS steers away from repetition during search.
        """
        if simulations is None:
            simulations = self.mcts_simulations
        if game_position_counts is None:
            game_position_counts = {}
        
        root = self.MCTSNode(root_board.copy(), parent=None, prior=0.0)
        move_probs, value = self.evaluate_batch([root.board], nodes=[root], model=model)[0]
        for move, prob in move_probs:
            child_board = root.board.copy()
            child_board.push(move)
            prior = float(prob) if np.isfinite(prob) and prob > 0.0 else 0.0
            root.children[move] = self.MCTSNode(child_board, parent=root, prior=prior)
        
        if add_dirichlet_noise and len(root.children) > 0:
            eps = self.mcts_dirichlet_eps
            alpha = self.mcts_dirichlet_alpha
            moves = list(root.children.keys())
            noise = np.random.dirichlet([alpha] * len(moves))
            for i, m in enumerate(moves):
                old_prior = root.children[m].prior
                if not np.isfinite(old_prior) or old_prior < 0.0:
                    old_prior = 0.0
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
                        q = -child.q_value
                        repeat_count = self.repetition_count_for_board(child.board, game_position_counts)
                        if repeat_count > 0:
                            q = float(np.clip(q + self.repetition_penalty_for_visits(repeat_count), -1.0, 1.0))
                        prior = child.prior if np.isfinite(child.prior) and child.prior > 0.0 else 0.0
                        u = self.mcts_c_puct * prior * math.sqrt(total_visits) / (1 + child.visits + child.virtual_loss_count)
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
            nodes_to_evaluate = []
            terminal_values = []
            terminal_draws = []
            eval_index_map = []
            
            for idx, node in enumerate(leaf_nodes):
                if self.is_terminal_for_training(node.board):
                    terminal_values.append(self.terminal_value_for_side_to_move(node.board))
                    terminal_draws.append(self.is_draw_result(node.board))
                else:
                    boards_to_evaluate.append(node.board)
                    nodes_to_evaluate.append(node)
                    terminal_values.append(None)
                    terminal_draws.append(False)
                    eval_index_map.append(idx)
            
            if boards_to_evaluate:
                eval_results = self.evaluate_batch(
                    boards_to_evaluate, nodes=nodes_to_evaluate, model=model
                )
            else:
                eval_results = []
            
            leaf_values = [None] * len(leaf_nodes)
            for eval_idx, node_idx in enumerate(eval_index_map):
                node = leaf_nodes[node_idx]
                mv_probs, leaf_value = eval_results[eval_idx]

                # Apply repetition penalty if this leaf position was already seen in the game.
                # Penalty scales with how many times it has been visited, same as play_game.
                prior_visits = self.repetition_count_for_board(node.board, game_position_counts)
                if prior_visits >= 1:
                    leaf_value = float(np.clip(
                        leaf_value + self.repetition_penalty_for_visits(prior_visits), -1.0, 1.0
                    ))

                leaf_values[node_idx] = leaf_value
                for move, prob in mv_probs:
                    if move not in node.children:
                        b = node.board.copy()
                        b.push(move)
                        prior = float(prob) if np.isfinite(prob) and prob > 0.0 else 0.0
                        node.children[move] = self.MCTSNode(b, parent=node, prior=prior)
            
            for idx, val in enumerate(terminal_values):
                if val is not None:
                    leaf_values[idx] = val
            
            for search_path, leaf_value, is_draw_leaf in zip(search_paths, leaf_values, terminal_draws):
                if leaf_value is None:
                    continue
                if is_draw_leaf:
                    for n in reversed(search_path):
                        n.visits += 1
                        n.value_sum += leaf_value
                        if n.virtual_loss_count > 0:
                            n.virtual_loss_count -= 1
                    continue
                value_to_propagate = leaf_value
                for n in reversed(search_path):
                    n.visits += 1
                    n.value_sum += value_to_propagate
                    if n.virtual_loss_count > 0:
                        n.virtual_loss_count -= 1
                    value_to_propagate = -value_to_propagate
        
        return root

    def run_mcts_parallel(self, root_boards, simulations=None, add_dirichlet_noise=False,
                          game_position_counts=None, model=None):
        """Run independent searches for several games while batching their leaves."""
        if simulations is None:
            simulations = self.mcts_simulations
        counts = game_position_counts or [{} for _ in root_boards]
        roots = [self.MCTSNode(board.copy()) for board in root_boards]
        initial = self.evaluate_batch([r.board for r in roots], nodes=roots, model=model)
        for root, (move_probs, _) in zip(roots, initial):
            for move, prob in move_probs:
                child_board = root.board.copy()
                child_board.push(move)
                root.children[move] = self.MCTSNode(
                    child_board, parent=root,
                    prior=float(prob) if np.isfinite(prob) and prob > 0 else 0.0)
            if add_dirichlet_noise and root.children:
                moves = list(root.children)
                noise = np.random.dirichlet([self.mcts_dirichlet_alpha] * len(moves))
                for move, n in zip(moves, noise):
                    root.children[move].prior = (
                        (1 - self.mcts_dirichlet_eps) * root.children[move].prior
                        + self.mcts_dirichlet_eps * n
                    )

        for start in range(0, simulations, self.mcts_batch_size):
            paths, leaves, owners = [], [], []
            batch_end = min(simulations, start + self.mcts_batch_size)
            for game_index, root in enumerate(roots):
                for _ in range(batch_end - start):
                    node, path = root, [root]
                    while node.children:
                        total = sum(c.visits + c.virtual_loss_count
                                    for c in node.children.values()) + 1
                        best_move, best_score = None, -1e9
                        for move, child in node.children.items():
                            q = -child.q_value
                            repeats = self.repetition_count_for_board(child.board, counts[game_index])
                            q = float(np.clip(q + self.repetition_penalty_for_visits(repeats), -1, 1))
                            prior = child.prior if np.isfinite(child.prior) and child.prior > 0 else 0.0
                            score = q + self.mcts_c_puct * prior * math.sqrt(total) / (
                                1 + child.visits + child.virtual_loss_count)
                            if score > best_score:
                                best_score, best_move = score, move
                        if best_move is None:
                            break
                        node = node.children[best_move]
                        node.virtual_loss_count += 1
                        path.append(node)
                    paths.append(path)
                    leaves.append(node)
                    owners.append(game_index)

            values, draw_flags, eval_nodes, eval_indices = [None] * len(leaves), [False] * len(leaves), [], []
            for index, node in enumerate(leaves):
                if self.is_terminal_for_training(node.board):
                    values[index] = self.terminal_value_for_side_to_move(node.board)
                    draw_flags[index] = self.is_draw_result(node.board)
                else:
                    eval_nodes.append(node)
                    eval_indices.append(index)
            if eval_nodes:
                results = self.evaluate_batch([n.board for n in eval_nodes], nodes=eval_nodes, model=model)
                for index, node, (move_probs, value) in zip(eval_indices, eval_nodes, results):
                    repeats = self.repetition_count_for_board(node.board, counts[owners[index]])
                    values[index] = float(np.clip(
                        value + self.repetition_penalty_for_visits(repeats), -1, 1))
                    for move, prob in move_probs:
                        if move not in node.children:
                            child_board = node.board.copy()
                            child_board.push(move)
                            node.children[move] = self.MCTSNode(
                                child_board, parent=node,
                                prior=float(prob) if np.isfinite(prob) and prob > 0 else 0.0)
            for path, value, is_draw in zip(paths, values, draw_flags):
                if value is None:
                    continue
                for node in reversed(path):
                    node.visits += 1
                    node.value_sum += value
                    node.virtual_loss_count = max(0, node.virtual_loss_count - 1)
                    if not is_draw:
                        value = -value
        return roots
    
    # -------------------------
    # Move selection
    # -------------------------
    def mcts_policy_from_root(self, root, temperature=1.0):
        if not root.children:
            return [], [], np.array([], dtype=np.float64)
        moves = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)
        visits = np.nan_to_num(visits, nan=0.0, posinf=0.0, neginf=0.0)
        visits = np.clip(visits, 0.0, None)
        if visits.sum() <= 0:
            visits = np.array([root.children[m].prior for m in moves], dtype=np.float64)
            visits = np.nan_to_num(visits, nan=0.0, posinf=0.0, neginf=0.0)
            visits = np.clip(visits, 0.0, None)
            visits = self.normalize_probabilities(visits)
        if temperature == 0 or temperature < 1e-8:
            probs = np.zeros_like(visits)
            probs[int(np.argmax(visits))] = 1.0
        else:
            probs = np.clip(visits, 1e-12, None) ** (1.0 / temperature)
            probs = self.normalize_probabilities(probs)
        return moves, visits, probs

    def select_move_with_policy(self, board, temperature=1.0, use_mcts=True,
                                add_dirichlet_noise=False, game_position_counts=None, model=None):
        flip = (board.turn == chess.BLACK)
        if not use_mcts:
            move_probs, _ = self.get_move_probabilities(board, model=model)
            if not move_probs:
                return None, ()
            moves, probs = zip(*move_probs)
            probs = self.normalize_probabilities(probs)
            selection_probs = self.avoid_repeated_position_probs(board, moves, probs, game_position_counts)
            selection_probs = self.normalize_probabilities(selection_probs)
            if temperature == 0 or temperature < 1e-8:
                selected_idx = int(np.argmax(selection_probs))
                train_probs = probs
            else:
                train_probs = probs ** (1.0 / temperature)
                train_probs = self.normalize_probabilities(train_probs)
                selected_idx = self.safe_choice_index(len(moves), selection_probs)
            policy_target = tuple(
                (self.move_to_index(move, flip=flip), float(prob))
                for move, prob in zip(moves, train_probs)
                if prob > 0.0
            )
            return moves[selected_idx], policy_target

        root = self.run_mcts_batched(board, simulations=self.mcts_simulations,
                                     add_dirichlet_noise=add_dirichlet_noise,
                                     game_position_counts=game_position_counts, model=model)
        self._last_root_q_value = root.q_value
        moves, _, probs = self.mcts_policy_from_root(root, temperature=temperature)
        if not moves:
            return None, ()
        mcts_policy_probs = probs
        selection_probs = self.avoid_repeated_position_probs(board, moves, mcts_policy_probs, game_position_counts)
        selection_probs = self.normalize_probabilities(selection_probs)
        selected_idx = self.safe_choice_index(len(moves), selection_probs)
        policy_target = tuple(
            (self.move_to_index(move, flip=flip), float(prob))
            for move, prob in zip(moves, mcts_policy_probs)
            if prob > 0.0
        )
        return moves[selected_idx], policy_target

    def select_move(self, board, temperature=1.0, use_mcts=True,
                    add_dirichlet_noise=False, game_position_counts=None, model=None):
        move, _ = self.select_move_with_policy(
            board,
            temperature=temperature,
            use_mcts=use_mcts,
            add_dirichlet_noise=add_dirichlet_noise,
            game_position_counts=game_position_counts,
            model=model
        )
        return move
    
    # -------------------------
    # Self-play with repetition-aware search
    # -------------------------
    def play_games_parallel(self, num_games=None, temperature=1.0, max_moves=200,
                            temp_threshold=30):
        """Play up to ``num_games`` self-play games, batching MCTS leaves."""
        total_games = self.parallel_games if num_games is None else max(0, int(num_games))
        if total_games == 0:
            return []
        active_limit = min(self.parallel_games, total_games)
        states = [{
            'board': chess.Board(), 'data': [], 'counts': self.new_repetition_tracker(),
            'move_count': 0, 'low_value_plies': 0, 'resigned': False, 'root_q': None
        } for _ in range(active_limit)]
        results = []
        games_started = active_limit
        while states and not self.stop_training_flag:
            search_states = []
            for state in states:
                board = state['board']
                if self.is_terminal_for_training(board) or state['move_count'] >= max_moves:
                    continue
                visit_count = self.repetition_count_for_board(board, state['counts'])
                self.record_position_visit(board, state['counts'])
                if visit_count > 0:
                    self.training_stats['repetition_penalties_applied'] += 1
                can_flip = self.is_position_symmetric_safe(board)
                board_tensor = self.board_to_tensor(board).cpu()
                legal_indices = self.legal_policy_indices(board)
                if state['move_count'] < self.opening_random_plies:
                    moves = list(board.legal_moves)
                    if moves:
                        move = random.choice(moves)
                        prob = 1.0 / len(moves)
                        policy = tuple((self.move_to_index(m, flip=board.turn == chess.BLACK), prob)
                                       for m in moves)
                        state['data'].append((board_tensor, policy, legal_indices, board.turn,
                                              can_flip, 0.0, None, 1.0))
                        board.push(move)
                        state['move_count'] += 1
                else:
                    search_states.append((state, board_tensor, legal_indices, can_flip))

            if search_states:
                roots = self.run_mcts_parallel(
                    [s[0]['board'] for s in search_states],
                    simulations=self.mcts_simulations,
                    add_dirichlet_noise=True,
                    game_position_counts=[s[0]['counts'] for s in search_states])
                for (state, board_tensor, legal_indices, can_flip), root in zip(search_states, roots):
                    self._last_root_q_value = root.q_value
                    state['root_q'] = root.q_value
                    state['low_value_plies'] = state['low_value_plies'] + 1 if (
                        root.q_value < self.resignation_threshold) else 0
                    if state['low_value_plies'] >= self.resignation_consecutive_plies:
                        state['resigned'] = True
                        self.training_stats['resigned_games'] += 1
                        continue
                    moves, _, probs = self.mcts_policy_from_root(root, temperature=(
                        temperature if state['move_count'] < temp_threshold else 0.0))
                    if not moves:
                        continue
                    selection = self.normalize_probabilities(
                        self.avoid_repeated_position_probs(
                            state['board'], moves, probs, state['counts']))
                    move = moves[self.safe_choice_index(len(moves), selection)]
                    policy = tuple((self.move_to_index(m, flip=state['board'].turn == chess.BLACK), float(p))
                                   for m, p in zip(moves, probs) if p > 0)
                    state['data'].append((board_tensor, policy, legal_indices, state['board'].turn,
                                          can_flip, 0.0, None, 1.0))
                    state['board'].push(move)
                    state['move_count'] += 1

            finished = [s for s in states if self.is_terminal_for_training(s['board']) or
                        s['move_count'] >= max_moves or s['resigned']]
            for state in finished:
                board, outcome = state['board'], state['board'].outcome(claim_draw=True)
                if state['resigned']:
                    reward = -1.0 if board.turn == chess.WHITE else 1.0
                elif outcome and outcome.winner == chess.WHITE:
                    reward = 1.0; self.training_stats['white_wins'] += 1
                elif outcome and outcome.winner == chess.BLACK:
                    reward = -1.0; self.training_stats['black_wins'] += 1
                else:
                    reward = self.draw_value_for_board(board)
                    self.training_stats['draws'] += 1
                    if state['move_count'] >= max_moves:
                        self.training_stats['max_move_draws'] += 1
                        self.training_stats['last_draw_reason'] = 'max moves'
                    else:
                        self.training_stats['rule_draws'] += 1
                        self.training_stats['last_draw_reason'] = (
                            outcome.termination.name.lower() if outcome else 'unknown')
                self.training_stats['games_played'] += 1
                self.training_stats['total_moves'] += len(state['data'])
                self.training_stats['last_game_moves'] = len(state['data'])
                state['result'] = (state['data'], reward)
                results.append(state['result'])
            finished_ids = {id(s) for s in finished}
            states = [s for s in states if id(s) not in finished_ids]
            # Recycle each completed slot immediately so the leaf-evaluation
            # batch remains full until the requested number of games starts.
            while len(states) < active_limit and games_started < total_games:
                states.append({
                    'board': chess.Board(), 'data': [],
                    'counts': self.new_repetition_tracker(), 'move_count': 0,
                    'low_value_plies': 0, 'resigned': False, 'root_q': None
                })
                games_started += 1
        for state in states:
            results.append((state['data'], self.draw_penalty))
        return results

    def play_game(self, temperature=1.0, max_moves=200, temp_threshold=30):
        """
        Play a self-play game.

        Repetition pressure is applied during MCTS selection only. Stored policy
        targets stay as MCTS visit counts, and value targets stay tied to the
        actual game result.
        """
        board = chess.Board()
        game_data = []
        move_count = 0
        self._last_root_q_value = None
        low_root_value_plies = 0
        resigned = False

        # Track exact positions and piece layouts. Layout repeats catch shuffling
        # cycles even when side-to-move, castling, or en-passant state differs.
        position_counts = self.new_repetition_tracker()

        while not self.is_terminal_for_training(board) and not self.stop_training_flag:
            visit_count = self.repetition_count_for_board(board, position_counts)
            self.record_position_visit(board, position_counts)
            if visit_count > 0:
                self.training_stats['repetition_penalties_applied'] += 1

            can_flip = self.is_position_symmetric_safe(board)
            board_tensor = self.board_to_tensor(board).cpu()
            legal_indices = self.legal_policy_indices(board)

            current_temp = temperature if move_count < temp_threshold else 0.0
            if move_count < self.opening_random_plies:
                legal_moves = list(board.legal_moves)
                if not legal_moves:
                    break
                move = random.choice(legal_moves)
                uniform_prob = 1.0 / len(legal_moves)
                policy_target = tuple(
                    (self.move_to_index(legal_move, flip=(board.turn == chess.BLACK)), uniform_prob)
                    for legal_move in legal_moves
                )
            else:
                move, policy_target = self.select_move_with_policy(
                    board,
                    temperature=current_temp,
                    use_mcts=True,
                    add_dirichlet_noise=True,
                    game_position_counts=position_counts
                )
                if self._last_root_q_value < self.resignation_threshold:
                    low_root_value_plies += 1
                else:
                    low_root_value_plies = 0
                if low_root_value_plies >= self.resignation_consecutive_plies:
                    resigned = True
                    self.training_stats['resigned_games'] += 1
                    break

            if move is None:
                break

            player = board.turn
            destination_penalty = self.move_into_repetition_penalty(board, move, position_counts)
            if destination_penalty < 0.0:
                self.training_stats['repetition_penalties_applied'] += 1

            game_data.append((board_tensor, policy_target, legal_indices, player, can_flip, 0.0, None, 1.0))
            board.push(move)
            move_count += 1

            if len(game_data) >= max_moves:
                break

        # Assign end-of-game reward
        outcome = board.outcome(claim_draw=True)
        if resigned:
            reward = -1.0 if board.turn == chess.WHITE else 1.0
        elif outcome and outcome.winner == chess.WHITE:
            reward = 1.0
            self.training_stats['white_wins'] += 1
        elif outcome and outcome.winner == chess.BLACK:
            reward = -1.0
            self.training_stats['black_wins'] += 1
        else:
            reward = self.draw_value_for_board(board)
            self.training_stats['draws'] += 1
            if len(game_data) >= max_moves:
                self.training_stats['max_move_draws'] += 1
                self.training_stats['last_draw_reason'] = 'max moves'
            else:
                self.training_stats['rule_draws'] += 1
                self.training_stats['last_draw_reason'] = outcome.termination.name.lower() if outcome else 'unknown'

        self.training_stats['games_played'] += 1
        self.training_stats['total_moves'] += len(game_data)
        self.training_stats['last_game_moves'] = len(game_data)

        return game_data, reward

    def play_stockfish_teacher_game(self, temperature=1.0, max_moves=200, temp_threshold=30):
        """
        Generate one teacher game using Stockfish policy/value targets.

        The engine provides the training target; repetition penalties are used
        only when sampling the played move from the teacher distribution.
        """
        board = chess.Board()
        game_data = []
        move_count = 0
        low_root_value_plies = 0
        resigned = False
        position_counts = self.new_repetition_tracker()

        while not self.is_terminal_for_training(board) and not self.stop_training_flag:
            self.record_position_visit(board, position_counts)
            teacher = self.stockfish_policy_value(board)
            if teacher is None:
                self.training_stats['stockfish_unavailable_games'] = (
                    self.training_stats.get('stockfish_unavailable_games', 0) + 1
                )
                return self.play_game(temperature, max_moves=max_moves, temp_threshold=temp_threshold)

            move_probs, teacher_value = teacher
            # Teacher games retain Stockfish's opening policy; random openings apply to self-play only.
            if move_count >= self.opening_random_plies and teacher_value < self.resignation_threshold:
                low_root_value_plies += 1
            else:
                low_root_value_plies = 0
            if low_root_value_plies >= self.resignation_consecutive_plies:
                resigned = True
                self.training_stats['resigned_games'] += 1
                break
            moves, probs = zip(*move_probs)
            current_temp = temperature if move_count < temp_threshold else 0.0
            if current_temp == 0 or current_temp < 1e-8:
                selection_probs = np.zeros(len(moves), dtype=np.float64)
                selection_probs[int(np.argmax(probs))] = 1.0
            else:
                selection_probs = np.asarray(probs, dtype=np.float64) ** (1.0 / current_temp)
                selection_probs = self.normalize_probabilities(selection_probs)

            selection_probs = self.avoid_repeated_position_probs(board, moves, selection_probs, position_counts)
            selection_probs = self.normalize_probabilities(selection_probs)
            selected_idx = self.safe_choice_index(len(moves), selection_probs)
            move = moves[selected_idx]

            can_flip = self.is_position_symmetric_safe(board)
            board_tensor = self.board_to_tensor(board).cpu()
            legal_indices = self.legal_policy_indices(board)
            policy_target = self.move_probs_to_policy_target(board, move_probs)
            player = board.turn
            game_data.append((board_tensor, policy_target, legal_indices, player, can_flip, 0.0, teacher_value, 1.0))

            board.push(move)
            move_count += 1
            if len(game_data) >= max_moves:
                break

        outcome = board.outcome(claim_draw=True)
        if resigned:
            reward = -1.0 if board.turn == chess.WHITE else 1.0
        elif outcome and outcome.winner == chess.WHITE:
            reward = 1.0
            self.training_stats['white_wins'] += 1
        elif outcome and outcome.winner == chess.BLACK:
            reward = -1.0
            self.training_stats['black_wins'] += 1
        else:
            reward = self.draw_value_for_board(board)
            self.training_stats['draws'] += 1
            if len(game_data) >= max_moves:
                self.training_stats['max_move_draws'] += 1
                self.training_stats['last_draw_reason'] = 'max moves'
            else:
                self.training_stats['rule_draws'] += 1
                self.training_stats['last_draw_reason'] = outcome.termination.name.lower() if outcome else 'unknown'

        self.training_stats['games_played'] += 1
        self.training_stats['total_moves'] += len(game_data)
        self.training_stats['last_game_moves'] = len(game_data)
        self.training_stats['stockfish_games'] = self.training_stats.get('stockfish_games', 0) + 1
        self.training_stats['stockfish_positions'] = (
            self.training_stats.get('stockfish_positions', 0) + len(game_data)
        )

        return game_data, reward

    # -------------------------
    # Replay buffer
    # -------------------------
    def unpack_game_position(self, entry):
        board_tensor, policy_target, legal_indices, player, can_flip = entry[:5]
        inline_penalty = entry[5] if len(entry) > 5 else 0.0
        value_override = entry[6] if len(entry) > 6 else None
        policy_weight = entry[7] if len(entry) > 7 else 1.0
        return board_tensor, policy_target, legal_indices, player, can_flip, inline_penalty, value_override, policy_weight

    def initial_replay_priority(self, policy_target, target_value):
        policy_indices, policy_probs = policy_target
        if torch.is_tensor(policy_probs) and policy_probs.numel() > 1:
            probs = policy_probs.float().clamp_min(1e-12)
            entropy = float(-(probs * probs.log()).sum().item())
            max_entropy = math.log(float(policy_probs.numel()))
            sharpness = max(0.0, 1.0 - entropy / max(max_entropy, 1e-12))
        else:
            sharpness = 1.0
        return float(self.priority_epsilon + 1.0 + abs(float(target_value)) + sharpness)

    def unpack_replay_entry(self, entry):
        if len(entry) >= 7:
            board_tensor, policy_target, legal_indices, target_value, age_marker, priority, policy_weight = entry[:7]
            return board_tensor, policy_target, legal_indices, target_value, age_marker, float(priority), float(policy_weight)
        if len(entry) == 6:
            board_tensor, policy_target, legal_indices, target_value, age_marker, priority = entry
            return board_tensor, policy_target, legal_indices, target_value, age_marker, float(priority), 1.0
        board_tensor, policy_target, legal_indices, target_value, age_marker = entry
        priority = self.initial_replay_priority(policy_target, target_value)
        return board_tensor, policy_target, legal_indices, target_value, age_marker, priority, 1.0

    def make_replay_entry(self, board_tensor, policy_target, legal_indices, target_value, age_marker, priority, policy_weight):
        return (
            board_tensor,
            policy_target,
            legal_indices,
            float(target_value),
            age_marker,
            float(max(priority, self.priority_epsilon)),
            float(np.clip(policy_weight, 0.0, 1.0)),
        )

    def sample_replay_indices(self):
        size = len(self.replay_buffer)
        if size == 0:
            return []

        sample_count = self.batch_size
        replace = size < sample_count
        if self.prioritized_replay_alpha <= 0:
            if replace:
                return [random.randrange(size) for _ in range(sample_count)]
            return random.sample(range(size), sample_count)

        priorities = []
        for entry in self.replay_buffer:
            *_, priority, _ = self.unpack_replay_entry(entry)
            priorities.append(max(float(priority), self.priority_epsilon))
        weights = np.asarray(priorities, dtype=np.float64) ** self.prioritized_replay_alpha
        probs = self.normalize_probabilities(weights)
        return np.random.choice(size, size=sample_count, replace=replace, p=probs).astype(int).tolist()

    def update_replay_priorities(self, sampled_indices, sample_errors):
        if sample_errors is None:
            return
        for idx, error in zip(sampled_indices, sample_errors):
            if idx < 0 or idx >= len(self.replay_buffer):
                continue
            board_tensor, policy_target, legal_indices, target_value, age_marker, _, policy_weight = self.unpack_replay_entry(
                self.replay_buffer[idx]
            )
            priority = float(abs(error)) + self.priority_epsilon
            self.replay_buffer[idx] = self.make_replay_entry(
                board_tensor,
                policy_target,
                legal_indices,
                target_value,
                age_marker,
                priority,
                policy_weight
            )

    def add_game_to_buffer(self, game_data, reward):
        """
        Add game to replay buffer.

        Value targets are the game outcome from the player's perspective unless
        a teacher value override is provided.
        """
        is_draw = self.is_draw_reward(reward)

        for position_entry in game_data:
            board_tensor_cpu, policy_target, legal_indices, player, can_flip, _inline_penalty, value_override, policy_weight = (
                self.unpack_game_position(position_entry)
            )
            if not policy_target:
                continue

            # Convert game outcome to this player's perspective
            if value_override is not None:
                target_value = float(np.clip(value_override, -1.0, 1.0))
            elif is_draw:
                base_value = reward
            elif player == chess.WHITE:
                base_value = reward
            else:
                base_value = -reward

            if value_override is None:
                target_value = float(np.clip(base_value, -1.0, 1.0))

            augmented = self.augment_training_entry(board_tensor_cpu, policy_target, legal_indices, can_flip)

            self.training_stats['positions_total'] += 1
            if can_flip:
                self.training_stats['positions_flipped'] += 1

            for aug_tensor, aug_policy_target, aug_legal_indices in augmented:
                compact_policy, compact_legal = self.compact_training_targets(
                    aug_policy_target,
                    aug_legal_indices
                )
                priority = self.initial_replay_priority(compact_policy, target_value)
                self.replay_buffer.append(self.make_replay_entry(
                    aug_tensor,
                    compact_policy,
                    compact_legal,
                    target_value,
                    self.data_counter,
                    priority,
                    policy_weight
                ))

            self.data_counter += 1
    
    def clean_old_data(self):
        if len(self.replay_buffer) < self.replay_capacity:
            return
        current_counter = self.data_counter
        new_buffer = deque(maxlen=self.replay_capacity)
        for entry in self.replay_buffer:
            board_tensor, policy_target, legal_indices, target_value, age_marker, priority, policy_weight = (
                self.unpack_replay_entry(entry)
            )
            if current_counter - age_marker < self.max_data_age:
                new_buffer.append(self.make_replay_entry(
                    board_tensor,
                    policy_target,
                    legal_indices,
                    target_value,
                    age_marker,
                    priority,
                    policy_weight
                ))
        self.replay_buffer = new_buffer
    
    def sample_batch(self):
        if len(self.replay_buffer) == 0:
            raise ValueError("Replay buffer is empty")
        sampled_indices = self.sample_replay_indices()
        batch = [self.unpack_replay_entry(self.replay_buffer[idx]) for idx in sampled_indices]
        boards, policy_targets, legal_indices_batch, target_values, _, _, policy_weights = zip(*batch)
        boards_tensor = torch.cat(boards, dim=0).to(self.device)
        if self.device.type == 'cuda':
            boards_tensor = boards_tensor.to(memory_format=torch.channels_last)

        batch_len = len(policy_targets)
        policy_targets_tensor = torch.zeros(
            (batch_len, 4096),
            dtype=torch.float32,
            device=self.device
        )
        legal_mask_tensor = torch.zeros(
            (batch_len, 4096),
            dtype=torch.bool,
            device=self.device
        )

        policy_row_parts = []
        policy_col_parts = []
        policy_val_parts = []
        legal_row_parts = []
        legal_col_parts = []

        for row, (policy_target, legal_indices) in enumerate(zip(policy_targets, legal_indices_batch)):
            # Older in-memory entries may still be tuple lists if training was already running.
            if (
                isinstance(policy_target, tuple)
                and len(policy_target) == 2
                and torch.is_tensor(policy_target[0])
            ):
                policy_indices, policy_probs = policy_target
            else:
                policy_target, legal_indices = self.compact_training_targets(
                    policy_target,
                    legal_indices
                )
                policy_indices, policy_probs = policy_target

            if not torch.is_tensor(legal_indices):
                _, legal_indices = self.compact_training_targets((), legal_indices)

            if policy_indices.numel() > 0:
                policy_row_parts.append(torch.full_like(policy_indices, row))
                policy_col_parts.append(policy_indices)
                policy_val_parts.append(policy_probs)
            if legal_indices.numel() > 0:
                legal_row_parts.append(torch.full_like(legal_indices, row))
                legal_col_parts.append(legal_indices)

        if policy_row_parts:
            policy_rows = torch.cat(policy_row_parts).to(self.device)
            policy_cols = torch.cat(policy_col_parts).to(self.device)
            policy_vals = torch.cat(policy_val_parts).to(self.device)
            policy_targets_tensor.index_put_(
                (policy_rows, policy_cols),
                policy_vals,
                accumulate=True
            )

        if legal_row_parts:
            legal_rows = torch.cat(legal_row_parts).to(self.device)
            legal_cols = torch.cat(legal_col_parts).to(self.device)
            legal_mask_tensor[legal_rows, legal_cols] = True

        row_sums = policy_targets_tensor.sum(dim=1, keepdim=True)
        policy_targets_tensor = torch.where(
            row_sums > 0,
            policy_targets_tensor / row_sums.clamp_min(1e-12),
            policy_targets_tensor
        )
        target_values_tensor = torch.tensor(
            target_values,
            dtype=torch.float32,
            device=self.device
        )
        policy_weights_tensor = torch.tensor(
            policy_weights,
            dtype=torch.float32,
            device=self.device
        )
        return boards_tensor, policy_targets_tensor, legal_mask_tensor, target_values_tensor, policy_weights_tensor, sampled_indices
    
    # -------------------------
    # Training
    # -------------------------
    def train_on_batch(self, boards_tensor, policy_targets_tensor, legal_mask_tensor,
                       target_values_tensor, policy_weights_tensor=None):
        self.model.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            policy_logits, values = self.model(boards_tensor)
            policy_logits = torch.nan_to_num(
                policy_logits.float(),
                nan=0.0,
                posinf=1e4,
                neginf=-1e4
            )
            values = torch.nan_to_num(
                values.float(),
                nan=0.0,
                posinf=1.0,
                neginf=-1.0
            )
            masked_logits = policy_logits.masked_fill(
                ~legal_mask_tensor,
                self.invalid_policy_logit(policy_logits)
            )
            log_probs = F.log_softmax(masked_logits, dim=1)
            per_policy_loss = -(policy_targets_tensor * log_probs).sum(dim=1)
            if policy_weights_tensor is None:
                policy_weights_tensor = torch.ones_like(per_policy_loss)
            policy_weights_tensor = policy_weights_tensor.clamp(0.0, 1.0)
            policy_loss = (
                per_policy_loss * policy_weights_tensor
            ).sum() / policy_weights_tensor.sum().clamp_min(1e-6)
            per_value_loss = F.mse_loss(values, target_values_tensor, reduction='none')
            value_loss = per_value_loss.mean()
            probs = F.softmax(masked_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()
            adaptive_entropy_coef = self.entropy_coef * (1.0 + 0.1 / (1.0 + self.training_stats['total_training_steps'] / 1000.0))
            loss = policy_loss + self.value_coef * value_loss - adaptive_entropy_coef * entropy
            sample_errors = (
                per_policy_loss.detach() * policy_weights_tensor.detach()
                + self.value_coef * per_value_loss.detach()
            )
        
        if not torch.isfinite(loss):
            return float('nan'), float('nan'), None
        
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
        
        return policy_loss.item(), value_loss.item(), sample_errors.float().cpu().numpy()

    def train_from_replay(self, steps=None, require_min_buffer=True):
        if steps is None:
            steps = self.train_steps_per_game
        if steps <= 0 or len(self.replay_buffer) == 0:
            return 0.0, 0.0, 0, 0.0
        if require_min_buffer and len(self.replay_buffer) < max(self.min_buffer_size, self.batch_size):
            return 0.0, 0.0, 0, 0.0

        avg_policy_loss = 0.0
        avg_value_loss = 0.0
        steps_done = 0
        train_start = time.time()

        for _ in range(steps):
            try:
                boards, policy_targets, legal_mask, values, policy_weights, sampled_indices = self.sample_batch()
            except ValueError:
                break
            p_loss, v_loss, sample_errors = self.train_on_batch(
                boards,
                policy_targets,
                legal_mask,
                values,
                policy_weights
            )
            if p_loss != p_loss or v_loss != v_loss:
                continue
            self.update_replay_priorities(sampled_indices, sample_errors)
            avg_policy_loss += p_loss
            avg_value_loss += v_loss
            steps_done += 1

        train_time = time.time() - train_start
        if steps_done > 0:
            avg_policy_loss /= steps_done
            avg_value_loss /= steps_done
        return avg_policy_loss, avg_value_loss, steps_done, train_time

    def evaluate_against_previous_checkpoint(self, games=None):
        games = self.evaluation_games if games is None else max(0, int(games))
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        if games <= 0 or not os.path.exists(model_path):
            return None

        checkpoint = torch.load(model_path, map_location='cpu')
        previous_state = checkpoint.get('model_state_dict')
        if not previous_state:
            return None
        previous_model = ChessNet().to(self.device)
        previous_model.load_state_dict(previous_state)
        if self.device.type == 'cuda':
            previous_model = previous_model.to(memory_format=torch.channels_last)
        previous_model.eval()
        current_model = self.model
        current_model.eval()
        results = {'wins': 0, 'losses': 0, 'draws': 0}
        replay_games = []
        try:
            for game_index in range(games):
                board = chess.Board()
                position_counts = self.new_repetition_tracker()
                replay = {'moves': [], 'evaluations': [], 'positions': [board.fen()]}
                while not self.is_terminal_for_training(board) and len(board.move_stack) < 200:
                    model = current_model if board.turn == chess.WHITE else previous_model
                    move = self.select_move(
                        board, temperature=0.0, use_mcts=True,
                        add_dirichlet_noise=False, game_position_counts=position_counts,
                        model=model
                    )
                    if move is None:
                        break
                    replay['moves'].append(move.uci())
                    replay['evaluations'].append(float(getattr(self, '_last_root_q_value', 0.0)))
                    self.record_position_visit(board, position_counts)
                    board.push(move)
                    replay['positions'].append(board.fen())

                outcome = board.outcome(claim_draw=True)
                if outcome is None or outcome.winner is None:
                    results['draws'] += 1
                elif outcome.winner == chess.WHITE:
                    results['wins'] += 1
                else:
                    results['losses'] += 1
                replay['result'] = (
                    'Draw' if outcome is None or outcome.winner is None
                    else ('White wins' if outcome.winner == chess.WHITE else 'Black wins')
                )
                replay_games.append(replay)
        finally:
            current_model.train()

        self.training_stats['evaluation_games'] += games
        self.training_stats['evaluation_wins'] += results['wins']
        self.training_stats['evaluation_losses'] += results['losses']
        self.training_stats['evaluation_draws'] += results['draws']
        self.last_evaluation_replays = replay_games
        self.evaluation_replay_version += 1
        return results
    
    def train(self, num_games=10, temperature=1.0, temp_threshold=30, callback=None):
        self.stop_training_flag = False
        game_num = 0
        while game_num < num_games and not self.stop_training_flag:
            batch_size = min(self.parallel_games, num_games - game_num)
            game_start = time.time()
            if self.should_use_stockfish_teacher():
                batch = [self.play_stockfish_teacher_game(
                    temperature, temp_threshold=temp_threshold)]
            else:
                batch = self.play_games_parallel(
                    batch_size, temperature=temperature, temp_threshold=temp_threshold)
            elapsed = time.time() - game_start
            per_game_time = elapsed / max(1, len(batch))
            for game_data, reward in batch:
                if game_num >= num_games:
                    break
                self.add_game_to_buffer(game_data, reward)
                if game_num % 10 == 0:
                    self.clean_old_data()
                p_loss, v_loss, steps_done, train_time = self.train_from_replay(
                    steps=self.train_steps_per_game, require_min_buffer=True)
                self.training_stats['total_game_time'] += per_game_time
                self.training_stats['total_train_time'] += train_time
                self.training_stats['last_game_time'] = per_game_time
                self.training_stats['last_train_time'] = train_time
                game_num += 1
                if callback:
                    callback(game_num, num_games, p_loss, v_loss, reward, per_game_time, train_time)
                if game_num % self.evaluation_interval == 0:
                    self.evaluate_against_previous_checkpoint()
                if game_num % 10 == 0:
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
                saved_state = checkpoint['model_state_dict']
                current_state = self.model.state_dict()
                incompatible = [
                    key for key, value in saved_state.items()
                    if key in current_state and current_state[key].shape != value.shape
                ]
                missing = [key for key in current_state.keys() if key not in saved_state]
                unexpected = [key for key in saved_state.keys() if key not in current_state]
                if incompatible or missing or unexpected:
                    reason = (
                        incompatible[0] if incompatible else
                        missing[0] if missing else
                        unexpected[0]
                    )
                    print(f"Saved model is incompatible with current architecture ({reason}). Starting fresh.")
                    return
                self.model.load_state_dict(saved_state)
                self.model.to(self.device)
                if self.device.type == 'cuda':
                    self.model = self.model.to(memory_format=torch.channels_last)
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                # Backwards compatibility
                for key, default in self.new_training_stats().items():
                    if key not in self.training_stats:
                        self.training_stats[key] = default
                if 'optimizer_state_dict' in checkpoint:
                    try:
                        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    except ValueError:
                        print("Optimizer state is incompatible. Using a fresh optimizer.")
                if 'scheduler_state_dict' in checkpoint:
                    try:
                        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                    except Exception:
                        print("Scheduler state is incompatible. Using a fresh scheduler.")
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
        self.window.title("Chess AI - Stockfish Teacher")
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
        self.last_move = None
        self.ai_thinking = False
        self.flip_board = False
        self.flip_var = tk.BooleanVar(value=False)
        self.message_queue = queue.Queue()
        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False
        self.player_vs_player_mode = False
        self.learn_from_human_var = tk.BooleanVar(value=True)
        self.human_game_data = []
        self.human_position_counts = self.ai.new_repetition_tracker()
        self.human_game_start_time = None
        self.evaluation_replays = []
        self.replay_game_index = 0
        self.replay_ply = 0
        self.replay_board = chess.Board()
        self.replay_playing = False
        self.replay_mode = False
        self.replay_delay_ms = 400
        self._seen_evaluation_replays = 0
        
        self.setup_gui()
        self.process_queue()
        
    def configure_style(self):
        self.colors = {
            'app_bg': '#ece7dc',
            'panel_bg': '#f7f3ea',
            'panel_border': '#cfc5b4',
            'text': '#1f2620',
            'muted': '#667164',
            'accent': '#2d6a4f',
            'accent_dark': '#1b4332',
            'danger': '#9d2f2f',
            'board_edge': '#3d3a32',
            'light_square': '#e8d8b8',
            'dark_square': '#769656',
            'selected_square': '#f5d76e',
            'last_move': '#b9c86b',
            'legal_move': '#244f35',
            'check': '#d85c4a',
            'text_bg': '#fffdf7',
        }
        self.window.configure(bg=self.colors['app_bg'])
        style = ttk.Style(self.window)
        try:
            style.theme_use('clam')
        except tk.TclError:
            pass

        base_font = ('Segoe UI', 10)
        heading_font = ('Segoe UI', 12, 'bold')
        title_font = ('Segoe UI', 18, 'bold')
        self.window.option_add('*Font', base_font)

        style.configure('TFrame', background=self.colors['panel_bg'])
        style.configure('App.TFrame', background=self.colors['app_bg'])
        style.configure('Panel.TFrame', background=self.colors['panel_bg'])
        style.configure('Header.TFrame', background=self.colors['accent_dark'])
        style.configure('Card.TLabelframe', background=self.colors['panel_bg'])
        style.configure('Card.TLabelframe.Label', background=self.colors['panel_bg'], foreground=self.colors['text'], font=heading_font)
        style.configure('TLabel', background=self.colors['panel_bg'], foreground=self.colors['text'])
        style.configure('Header.TLabel', background=self.colors['accent_dark'], foreground='#fffdf7', font=title_font)
        style.configure('HeaderMeta.TLabel', background=self.colors['accent_dark'], foreground='#dce8d7')
        style.configure('Muted.TLabel', background=self.colors['panel_bg'], foreground=self.colors['muted'])
        style.configure('Value.TLabel', background=self.colors['panel_bg'], foreground=self.colors['accent_dark'], font=('Segoe UI', 10, 'bold'))
        style.configure('Status.TLabel', background='#ded7c8', foreground=self.colors['text'], padding=(10, 6))
        style.configure('TEntry', fieldbackground='#fffdf7', foreground=self.colors['text'], padding=4)
        style.configure('TButton', padding=(10, 6))
        style.configure('Accent.TButton', background=self.colors['accent'], foreground='white')
        style.map('Accent.TButton', background=[('active', self.colors['accent_dark']), ('disabled', '#94a99a')])
        style.configure('Danger.TButton', background=self.colors['danger'], foreground='white')
        style.map('Danger.TButton', background=[('active', '#7d2424'), ('disabled', '#c9aaaa')])
        style.configure('TNotebook', background=self.colors['app_bg'], borderwidth=0)
        style.configure('TNotebook.Tab', padding=(14, 7), font=('Segoe UI', 10, 'bold'))

    def setup_gui(self):
        self.configure_style()
        self.window.title("Chess AI Self-Play Lab")
        self.window.geometry("1180x840")
        self.window.minsize(1040, 720)
        self.window.columnconfigure(0, weight=1)
        self.window.rowconfigure(0, weight=1)

        main_frame = ttk.Frame(self.window, padding=14, style='App.TFrame')
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        main_frame.columnconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=0)
        main_frame.rowconfigure(1, weight=1)

        header_frame = ttk.Frame(main_frame, padding=(16, 12), style='Header.TFrame')
        header_frame.grid(row=0, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 12))
        header_frame.columnconfigure(0, weight=1)
        ttk.Label(header_frame, text="Chess AI Self-Play Lab", style='Header.TLabel').grid(row=0, column=0, sticky=tk.W)
        self.summary_var = tk.StringVar(value="Ready")
        ttk.Label(header_frame, textvariable=self.summary_var, style='HeaderMeta.TLabel').grid(row=1, column=0, sticky=tk.W, pady=(3, 0))
        self.device_var = tk.StringVar(value=f"{self.ai.device} | AMP {'ON' if self.ai.use_amp else 'OFF'}")
        ttk.Label(header_frame, textvariable=self.device_var, style='HeaderMeta.TLabel').grid(row=0, column=1, rowspan=2, sticky=tk.E)
        
        left_frame = ttk.Frame(main_frame, style='App.TFrame')
        left_frame.grid(row=1, column=0, sticky=(tk.N, tk.S, tk.W), padx=(0, 14))
        
        board_container = tk.Frame(left_frame, bg=self.colors['board_edge'], padx=10, pady=10)
        board_container.grid(row=0, column=0, sticky=tk.N, pady=(0, 12))
        
        self.board_label = tk.Label(board_container, bg=self.colors['board_edge'], bd=0, cursor='hand2')
        self.board_label.grid(row=0, column=0)
        self.board_label.bind("<Button-1>", self.on_board_click)
        
        history_frame = ttk.LabelFrame(left_frame, text="Move History", padding=10, style='Card.TLabelframe')
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        history_frame.columnconfigure(0, weight=1)
        
        self.history_text = scrolledtext.ScrolledText(
            history_frame, height=8, width=54, wrap=tk.WORD, relief=tk.FLAT,
            bg=self.colors['text_bg'], fg=self.colors['text'], insertbackground=self.colors['text'],
            padx=10, pady=8
        )
        self.history_text.grid(row=0, column=0, sticky=(tk.W, tk.E))
        ttk.Button(history_frame, text="Copy Moves", command=self.copy_moves).grid(row=1, column=0, sticky=tk.E, pady=(8, 0))
        
        right_frame = ttk.Frame(main_frame, style='App.TFrame')
        right_frame.grid(row=1, column=1, sticky=(tk.N, tk.S, tk.W, tk.E))
        right_frame.columnconfigure(0, weight=1)
        right_frame.rowconfigure(0, weight=1)
        
        notebook = ttk.Notebook(right_frame)
        notebook.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))

        train_tab = ttk.Frame(notebook, padding=10, style='Panel.TFrame')
        play_tab = ttk.Frame(notebook, padding=10, style='Panel.TFrame')
        stats_tab = ttk.Frame(notebook, padding=10, style='Panel.TFrame')
        notebook.add(train_tab, text="Training")
        notebook.add(play_tab, text="Play")
        notebook.add(stats_tab, text="Stats")

        train_frame = ttk.LabelFrame(train_tab, text="Self-Play Parameters", padding=12, style='Card.TLabelframe')
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        train_frame.columnconfigure(1, weight=1)

        def add_entry(row, label, variable, hint=None):
            ttk.Label(train_frame, text=label).grid(row=row, column=0, sticky=tk.W, pady=6, padx=(0, 10))
            entry = ttk.Entry(train_frame, textvariable=variable, width=16)
            entry.grid(row=row, column=1, sticky=(tk.W, tk.E), pady=6)
            if hint:
                ttk.Label(train_frame, text=hint, style='Muted.TLabel').grid(row=row, column=2, sticky=tk.W, padx=(10, 0))
            return entry
        
        self.num_games_var = tk.StringVar(value="10")
        self.temperature_var = tk.StringVar(value="1.0")
        self.temp_threshold_var = tk.StringVar(value="30")
        self.mcts_sims_var = tk.StringVar(value=str(self.ai.mcts_simulations))
        self.mcts_batch_var = tk.StringVar(value=str(self.ai.mcts_batch_size))
        self.train_steps_var = tk.StringVar(value=str(self.ai.train_steps_per_game))
        self.draw_penalty_var = tk.StringVar(value=str(self.ai.draw_penalty))
        self.repeat_penalty_var = tk.StringVar(value=str(self.ai.repetition_penalty))
        self.rep_draw_penalty_var = tk.StringVar(value=str(self.ai.repetition_draw_penalty))
        self.stockfish_path_var = tk.StringVar(value=self.ai.stockfish_path or "")
        self.stockfish_start_var = tk.StringVar(value=str(self.ai.stockfish_teacher_start))
        self.stockfish_end_var = tk.StringVar(value=str(self.ai.stockfish_teacher_end))
        self.stockfish_decay_var = tk.StringVar(value=str(self.ai.stockfish_teacher_decay_games))
        self.resignation_threshold_var = tk.StringVar(value=str(self.ai.resignation_threshold))
        self.resignation_plies_var = tk.StringVar(value=str(self.ai.resignation_consecutive_plies))
        self.opening_random_plies_var = tk.StringVar(value=str(self.ai.opening_random_plies))
        self.evaluation_games_var = tk.StringVar(value=str(self.ai.evaluation_games))
        self.evaluation_interval_var = tk.StringVar(value=str(self.ai.evaluation_interval))
        self.parallel_games_var = tk.StringVar(value=str(self.ai.parallel_games))

        add_entry(0, "Games", self.num_games_var)
        add_entry(1, "Temperature", self.temperature_var)
        add_entry(2, "Explore moves", self.temp_threshold_var, "then greedy")
        add_entry(3, "MCTS simulations", self.mcts_sims_var)
        add_entry(4, "MCTS batch", self.mcts_batch_var)
        add_entry(5, "Train steps/game", self.train_steps_var)
        add_entry(6, "Draw penalty", self.draw_penalty_var)
        add_entry(7, "Repeat penalty", self.repeat_penalty_var)
        add_entry(8, "Repeat draw penalty", self.rep_draw_penalty_var)
        add_entry(9, "Stockfish path", self.stockfish_path_var, "optional")
        add_entry(10, "Teacher start", self.stockfish_start_var)
        add_entry(11, "Teacher end", self.stockfish_end_var)
        add_entry(12, "Teacher decay games", self.stockfish_decay_var)
        add_entry(13, "Resignation threshold", self.resignation_threshold_var)
        add_entry(14, "Resignation plies", self.resignation_plies_var)
        add_entry(15, "Random opening plies", self.opening_random_plies_var)
        add_entry(16, "Evaluation games", self.evaluation_games_var)
        add_entry(17, "Evaluation interval", self.evaluation_interval_var, "training games")
        add_entry(18, "Parallel games", self.parallel_games_var)
        
        button_frame = ttk.Frame(train_frame)
        button_frame.grid(row=19, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=(12, 4))
        for col in range(3):
            button_frame.columnconfigure(col, weight=1)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training, style='Accent.TButton')
        self.train_button.grid(row=0, column=0, sticky=(tk.W, tk.E), padx=(0, 6))
        
        self.stop_button = ttk.Button(button_frame, text="Stop", command=self.stop_training, state=tk.DISABLED, style='Danger.TButton')
        self.stop_button.grid(row=0, column=1, sticky=(tk.W, tk.E), padx=6)

        self.reset_model_button = ttk.Button(button_frame, text="Fresh Start", command=self.reset_model)
        self.reset_model_button.grid(row=0, column=2, sticky=(tk.W, tk.E), padx=(6, 0))
        
        progress_frame = ttk.LabelFrame(train_tab, text="Training Status", padding=12, style='Card.TLabelframe')
        progress_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(12, 0))
        self.progress_var = tk.StringVar(value="No training in progress")
        ttk.Label(progress_frame, textvariable=self.progress_var, wraplength=360, style='Value.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        play_frame = ttk.LabelFrame(play_tab, text="Game Controls", padding=12, style='Card.TLabelframe')
        play_frame.grid(row=0, column=0, sticky=(tk.W, tk.E))
        for col in range(2):
            play_frame.columnconfigure(col, weight=1)
        
        ttk.Button(play_frame, text="Play White", command=lambda: self.start_game(chess.WHITE), style='Accent.TButton').grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        ttk.Button(play_frame, text="Play Black", command=lambda: self.start_game(chess.BLACK), style='Accent.TButton').grid(row=0, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        ttk.Button(play_frame, text="Player vs Player", command=self.start_player_vs_player).grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(play_frame, text="AI vs AI Demo", command=self.watch_ai_game).grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        self.pause_button = ttk.Button(play_frame, text="Pause", command=self.toggle_pause_ai_game, state=tk.DISABLED)
        self.pause_button.grid(row=3, column=0, sticky=(tk.W, tk.E), pady=5, padx=(0, 5))
        ttk.Button(play_frame, text="New Game", command=self.reset_game).grid(row=3, column=1, sticky=(tk.W, tk.E), pady=5, padx=(5, 0))
        ttk.Checkbutton(play_frame, text="Flip board", variable=self.flip_var, command=self.on_flip_toggle).grid(row=4, column=0, sticky=tk.W, pady=(12, 4))
        ttk.Checkbutton(play_frame, text="Learn from my games", variable=self.learn_from_human_var).grid(row=5, column=0, columnspan=2, sticky=tk.W, pady=4)
        replay_frame = ttk.LabelFrame(play_tab, text="Evaluation Replay", padding=12, style='Card.TLabelframe')
        replay_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=(12, 0))
        self.replay_game_var = tk.StringVar(value="No evaluation games")
        self.replay_info_var = tk.StringVar(value="Run an evaluation cycle to record games.")
        ttk.Label(replay_frame, textvariable=self.replay_game_var).grid(row=0, column=0, columnspan=4, sticky=tk.W)
        ttk.Label(replay_frame, textvariable=self.replay_info_var, style='Muted.TLabel').grid(row=1, column=0, columnspan=4, sticky=tk.W, pady=(4, 8))
        ttk.Button(replay_frame, text="Game -", command=lambda: self.select_replay_game(-1)).grid(row=2, column=0, sticky=(tk.W, tk.E), padx=(0, 4))
        ttk.Button(replay_frame, text="Game +", command=lambda: self.select_replay_game(1)).grid(row=2, column=1, sticky=(tk.W, tk.E), padx=4)
        ttk.Button(replay_frame, text="Restart", command=self.restart_replay).grid(row=2, column=2, sticky=(tk.W, tk.E), padx=4)
        self.replay_pause_button = ttk.Button(replay_frame, text="Play", command=self.toggle_replay)
        self.replay_pause_button.grid(row=2, column=3, sticky=(tk.W, tk.E), padx=(4, 0))
        ttk.Button(replay_frame, text="Previous", command=lambda: self.step_replay(-1)).grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(6, 0), padx=(0, 4))
        ttk.Button(replay_frame, text="Next", command=lambda: self.step_replay(1)).grid(row=3, column=2, columnspan=2, sticky=(tk.W, tk.E), pady=(6, 0), padx=(4, 0))
        
        stats_frame = ttk.LabelFrame(stats_tab, text="Training Statistics", padding=12, style='Card.TLabelframe')
        stats_frame.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        stats_tab.rowconfigure(0, weight=1)
        stats_tab.columnconfigure(0, weight=1)
        stats_frame.rowconfigure(0, weight=1)
        stats_frame.columnconfigure(0, weight=1)
        
        self.stats_text = tk.Text(
            stats_frame, height=29, width=42, wrap=tk.WORD, relief=tk.FLAT,
            bg=self.colors['text_bg'], fg=self.colors['text'], insertbackground=self.colors['text'],
            padx=10, pady=8
        )
        self.stats_text.grid(row=0, column=0, sticky=(tk.N, tk.S, tk.W, tk.E))
        ttk.Button(stats_frame, text="Copy Stats", command=self.copy_stats).grid(row=1, column=0, sticky=tk.E, pady=(8, 0))

        badge_frame = ttk.Frame(stats_tab, padding=(0, 10, 0, 0), style='Panel.TFrame')
        badge_frame.grid(row=1, column=0, sticky=(tk.W, tk.E))
        ttk.Label(badge_frame, text="Teacher + repetition-aware search active", style='Value.TLabel').grid(row=0, column=0, sticky=tk.W)
        
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, style='Status.TLabel', anchor=tk.W).grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(12, 0))
        
        self.update_board_display()
        self.update_stats_display()
        self.update_summary_display()
    
    def on_flip_toggle(self):
        self.flip_board = bool(self.flip_var.get())
        self.update_board_display()
    
    def legacy_board_to_image(self):
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
        
        offset = 32
        
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
                if square == self.selected_square and not self.replay_replays_active():
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
    
    def board_to_image(self):
        display_board = self.replay_board if self.replay_replays_active() else self.board
        display_last_move = self.replay_last_move()
        board_size = self.square_size * 8
        offset = 32
        margin = offset * 2
        image = Image.new('RGB', (board_size + margin, board_size + margin), self.colors['app_bg'])
        draw = ImageDraw.Draw(image)

        try:
            piece_font = ImageFont.truetype("seguisym.ttf", int(self.square_size * 0.7))
            coord_font = ImageFont.truetype("arial.ttf", 13)
        except:
            try:
                piece_font = ImageFont.truetype("Arial.ttf", int(self.square_size * 0.7))
                coord_font = ImageFont.truetype("Arial.ttf", 13)
            except:
                piece_font = ImageFont.load_default()
                coord_font = ImageFont.load_default()

        draw.rounded_rectangle(
            [offset - 6, offset - 6, offset + board_size + 6, offset + board_size + 6],
            radius=12,
            fill=self.colors['board_edge']
        )

        legal_targets = (
            {move.to_square for move in self.legal_moves_for_selected}
            if not self.replay_replays_active() else set()
        )
        last_move_squares = set()
        if display_last_move:
            last_move_squares = {display_last_move.from_square, display_last_move.to_square}
        check_square = display_board.king(display_board.turn) if display_board.is_check() else None

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
                    color = self.colors['selected_square']
                elif square == check_square:
                    color = self.colors['check']
                elif square in last_move_squares:
                    color = self.colors['last_move']
                elif (rank + file) % 2 == 0:
                    color = self.colors['light_square']
                else:
                    color = self.colors['dark_square']

                draw.rectangle([x1, y1, x2, y2], fill=color)
                if square in legal_targets:
                    center = (x1 + self.square_size // 2, y1 + self.square_size // 2)
                    if display_board.piece_at(square):
                        inset = 8
                        draw.ellipse(
                            [x1 + inset, y1 + inset, x2 - inset, y2 - inset],
                            outline=self.colors['legal_move'],
                            width=4
                        )
                    else:
                        radius = max(5, self.square_size // 8)
                        draw.ellipse(
                            [center[0] - radius, center[1] - radius, center[0] + radius, center[1] + radius],
                            fill=self.colors['legal_move']
                        )

        files = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
        ranks = ['1', '2', '3', '4', '5', '6', '7', '8']
        if self.flip_board:
            files = list(reversed(files))
            ranks = list(reversed(ranks))

        for i, file_char in enumerate(files):
            x = i * self.square_size + self.square_size // 2 + offset
            draw.text((x, 14), file_char, fill=self.colors['muted'], font=coord_font, anchor='mm')
            draw.text((x, board_size + offset + 18), file_char, fill=self.colors['muted'], font=coord_font, anchor='mm')

        for i, rank_char in enumerate(ranks):
            y = ((7 - i) if not self.flip_board else i) * self.square_size + self.square_size // 2 + offset
            draw.text((14, y), rank_char, fill=self.colors['muted'], font=coord_font, anchor='mm')
            draw.text((board_size + offset + 18, y), rank_char, fill=self.colors['muted'], font=coord_font, anchor='mm')

        piece_symbols = {
            'P': '\u2659', 'N': '\u2658', 'B': '\u2657', 'R': '\u2656', 'Q': '\u2655', 'K': '\u2654',
            'p': '\u265f', 'n': '\u265e', 'b': '\u265d', 'r': '\u265c', 'q': '\u265b', 'k': '\u265a'
        }

        for rank in range(8):
            for file in range(8):
                square = chess.square(file, rank)
                piece = display_board.piece_at(square)
                if piece:
                    piece_char = piece_symbols.get(piece.symbol(), piece.symbol())
                    if not self.flip_board:
                        x = file * self.square_size + self.square_size // 2 + offset
                        y = (7 - rank) * self.square_size + self.square_size // 2 + offset
                    else:
                        x = (7 - file) * self.square_size + self.square_size // 2 + offset
                        y = rank * self.square_size + self.square_size // 2 + offset

                    piece_color = '#f8f4e8' if piece.color == chess.WHITE else '#1f2420'
                    outline_color = '#20251f' if piece.color == chess.WHITE else '#f8f4e8'
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
        self.update_summary_display()

    def current_mode_text(self):
        if self.is_training:
            return "Training"
        if self.ai_vs_ai_running:
            return "AI vs AI paused" if self.ai_vs_ai_paused else "AI vs AI"
        if self.player_vs_player_mode:
            return "Player vs Player"
        if self.human_color == chess.WHITE:
            return "Playing White"
        if self.human_color == chess.BLACK:
            return "Playing Black"
        return "Ready"

    def update_summary_display(self):
        if not hasattr(self, 'summary_var'):
            return
        turn = "White" if self.board.turn == chess.WHITE else "Black"
        status = "check" if self.board.is_check() else "to move"
        stats = self.ai.training_stats
        self.summary_var.set(
            f"{self.current_mode_text()} | {turn} {status} | "
            f"{len(self.move_history)} plies | {stats.get('games_played', 0)} trained games | "
            f"{stats.get('draws', 0)} draws"
        )
    
    def update_stats_display(self):
        stats = self.ai.training_stats
        win_rate_white = (stats['white_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        win_rate_black = (stats['black_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        draw_rate = (stats['draws'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        flip_rate = (stats['positions_flipped'] / stats['positions_total'] * 100) if stats['positions_total'] > 0 else 0
        current_lr = self.ai.optimizer.param_groups[0]['lr']
        rep_penalties = stats.get('repetition_penalties_applied', 0)
        avg_game_time = (stats.get('total_game_time', 0.0) / stats['games_played']) if stats['games_played'] > 0 else 0.0
        avg_train_time = (stats.get('total_train_time', 0.0) / stats['games_played']) if stats['games_played'] > 0 else 0.0
        avg_moves = (stats['total_moves'] / stats['games_played']) if stats['games_played'] > 0 else 0.0
        last_moves = stats.get('last_game_moves', 0)
        last_game_time = stats.get('last_game_time', 0.0)
        seconds_per_move = (last_game_time / last_moves) if last_moves > 0 else 0.0
        stockfish_status = "available" if self.ai.stockfish_available() else "off"
        if self.ai.stockfish_disabled_reason:
            stockfish_status = "disabled"
        
        stats_text = f"""Games: {stats['games_played']}
Moves: {stats['total_moves']}
Steps: {stats.get('total_training_steps', 0)}

White: {stats['white_wins']} ({win_rate_white:.1f}%)
Black: {stats['black_wins']} ({win_rate_black:.1f}%)
Draws: {stats['draws']} ({draw_rate:.1f}%)

Buffer: {len(self.ai.replay_buffer)}
Augmentation: {flip_rate:.1f}% flipped
Rep.penalties: {rep_penalties}
Human games: {stats.get('human_games', 0)}
Human positions: {stats.get('human_positions', 0)}
Human train steps: {stats.get('human_train_steps', 0)}
Stockfish games: {stats.get('stockfish_games', 0)}
Stockfish positions: {stats.get('stockfish_positions', 0)}
Stockfish skipped: {stats.get('stockfish_unavailable_games', 0)}
Resigned games: {stats.get('resigned_games', 0)}
Eval (current vs previous): {stats.get('evaluation_wins', 0)}-{stats.get('evaluation_losses', 0)}-{stats.get('evaluation_draws', 0)}
Last game: {last_moves} moves, {last_game_time:.1f}s
Avg game: {avg_game_time:.1f}s, {avg_moves:.1f} moves
Avg train: {avg_train_time:.2f}s
Sec/move: {seconds_per_move:.2f}
Rule draws: {stats.get('rule_draws', 0)}
Max-move draws: {stats.get('max_move_draws', 0)}
Last draw: {stats.get('last_draw_reason', '')}
LR: {current_lr:.2e}
MCTS sims: {self.ai.mcts_simulations}
MCTS batch: {self.ai.mcts_batch_size}
Steps/game: {self.ai.train_steps_per_game}
Draw penalty: {self.ai.draw_penalty}
Rep penalty: {self.ai.repetition_penalty}
Rep draw penalty: {self.ai.repetition_draw_penalty}
Teacher rate: {self.ai.stockfish_teacher_rate():.2f}
Stockfish: {stockfish_status}
Model: {self.ai.save_dir}""".strip()
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.update_summary_display()
    
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

    def reset_human_learning_game(self):
        self.human_game_data = []
        self.human_position_counts = self.ai.new_repetition_tracker()
        self.human_game_start_time = time.time()

    def human_move_policy_target(self, move):
        flip = (self.board.turn == chess.BLACK)
        move_idx = self.ai.move_to_index(move, flip=flip)
        if 0 <= move_idx < 4096:
            return ((move_idx, 1.0),)
        return ()

    def active_human_color(self):
        if self.player_vs_player_mode:
            return self.board.turn
        return self.human_color

    def record_human_learning_position(self, policy_target, move=None, verify_human=False):
        if not self.learn_from_human_var.get():
            return
        if self.human_color is None and not self.player_vs_player_mode:
            return
        if not policy_target:
            return

        policy_weight = 1.0
        if verify_human and move is not None:
            policy_target, policy_weight = self.ai.verified_human_policy_target(
                self.board,
                move,
                policy_target
            )

        visit_count = self.ai.repetition_count_for_board(self.board, self.human_position_counts)
        self.ai.record_position_visit(self.board, self.human_position_counts)
        if visit_count > 0:
            self.ai.training_stats['repetition_penalties_applied'] += 1
        destination_penalty = self.ai.move_into_repetition_penalty(
            self.board,
            move,
            self.human_position_counts
        )
        if destination_penalty < 0.0:
            self.ai.training_stats['repetition_penalties_applied'] += 1

        can_flip = self.ai.is_position_symmetric_safe(self.board)
        board_tensor = self.ai.board_to_tensor(self.board).cpu()
        legal_indices = self.ai.legal_policy_indices(self.board)
        player = self.board.turn
        self.human_game_data.append((
            board_tensor,
            policy_target,
            legal_indices,
            player,
            can_flip,
            0.0,
            None,
            policy_weight
        ))

    def finish_human_learning_game(self):
        if not self.learn_from_human_var.get() or not self.human_game_data:
            return ""

        outcome = self.board.outcome(claim_draw=True)
        if outcome and outcome.winner == chess.WHITE:
            reward = 1.0
            self.ai.training_stats['white_wins'] += 1
        elif outcome and outcome.winner == chess.BLACK:
            reward = -1.0
            self.ai.training_stats['black_wins'] += 1
        else:
            reward = self.ai.draw_value_for_board(self.board)
            self.ai.training_stats['draws'] += 1
            self.ai.training_stats['rule_draws'] += 1
            self.ai.training_stats['last_draw_reason'] = (
                outcome.termination.name.lower()
                if outcome else 'unknown'
            )

        game_time = time.time() - self.human_game_start_time if self.human_game_start_time else 0.0
        positions = len(self.human_game_data)

        self.ai.add_game_to_buffer(self.human_game_data, reward)
        self.ai.clean_old_data()
        p_loss, v_loss, steps_done, train_time = self.ai.train_from_replay(
            steps=self.ai.train_steps_per_game,
            require_min_buffer=False
        )

        stats = self.ai.training_stats
        stats['games_played'] += 1
        stats['total_moves'] += positions
        stats['last_game_moves'] = positions
        stats['total_game_time'] += game_time
        stats['total_train_time'] += train_time
        stats['last_game_time'] = game_time
        stats['last_train_time'] = train_time
        stats['human_games'] = stats.get('human_games', 0) + 1
        stats['human_positions'] = stats.get('human_positions', 0) + positions
        stats['human_train_steps'] = stats.get('human_train_steps', 0) + steps_done

        self.human_game_data = []
        self.human_position_counts = self.ai.new_repetition_tracker()
        self.human_game_start_time = None
        self.ai.save_model()
        self.update_stats_display()

        if steps_done > 0:
            return f"\nLearned from {positions} positions. P: {p_loss:.3f} V: {v_loss:.3f}"
        return f"\nSaved {positions} positions to replay."
    
    def on_board_click(self, event):
        if self.replay_replays_active():
            return
        active_color = self.active_human_color()
        if active_color is None or self.board.turn != active_color or self.ai.is_terminal_for_training(self.board) or self.ai_thinking:
            return
        offset = 32
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
            if piece and piece.color == active_color:
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
                self.record_human_learning_position(
                    self.human_move_policy_target(move),
                    move,
                    verify_human=True
                )
                self.make_move(move)
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.update_board_display()
                if self.ai.is_terminal_for_training(self.board):
                    self.game_over()
                elif self.player_vs_player_mode:
                    next_player = "White" if self.board.turn == chess.WHITE else "Black"
                    self.status_var.set(f"{next_player} to move")
                else:
                    self.window.after(300, self.ai_move)
            else:
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.status_var.set("Illegal move")
                self.update_board_display()
    
    def legacy_ask_promotion_piece(self):
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
    
    def ask_promotion_piece(self):
        dlg = tk.Toplevel(self.window)
        dlg.title("Promotion")
        dlg.transient(self.window)
        dlg.grab_set()
        dlg.configure(bg=self.colors['panel_bg'])
        choice = {'piece': None}

        ttk.Label(dlg, text="Promote to", padding=10, style='Value.TLabel').grid(row=0, column=0, columnspan=4)
        pieces = [
            ('\u2655', chess.QUEEN),
            ('\u2656', chess.ROOK),
            ('\u2657', chess.BISHOP),
            ('\u2658', chess.KNIGHT),
        ]
        for col, (label, piece_type) in enumerate(pieces):
            ttk.Button(
                dlg,
                text=label,
                width=6,
                command=lambda pt=piece_type: [choice.update({'piece': pt}), dlg.destroy()]
            ).grid(row=1, column=col, padx=6, pady=(0, 10))
        dlg.wait_window()
        return choice['piece']

    def make_move(self, move):
        san = self.board.san(move)
        self.board.push(move)
        self.last_move = move
        self.move_history.append(san)
        self.update_move_history()
    
    def ai_move(self):
        if self.ai.is_terminal_for_training(self.board):
            self.game_over()
            return
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        self.window.update()
        try:
            move, policy_target = self.ai.select_move_with_policy(
                self.board,
                temperature=0.0,
                use_mcts=True,
                add_dirichlet_noise=False
            )
            if move:
                self.record_human_learning_position(policy_target, move)
                self.make_move(move)
                self.status_var.set(f"AI: {self.move_history[-1]}")
                self.update_board_display()
                if self.ai.is_terminal_for_training(self.board):
                    self.game_over()
        except Exception as e:
            messagebox.showerror("Error", str(e))
        finally:
            self.ai_thinking = False
    
    def start_game(self, color):
        if self.is_training and self.learn_from_human_var.get():
            messagebox.showwarning("Warning", "Stop self-play training before starting a learning game")
            return
        self.replay_mode = False
        self.board = chess.Board()
        self.human_color = color
        self.player_vs_player_mode = False
        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.last_move = None
        self.ai_thinking = False
        self.flip_board = (color == chess.BLACK)
        self.flip_var.set(self.flip_board)
        self.reset_human_learning_game()
        self.update_board_display()
        self.update_move_history()
        self.status_var.set(f"You are {'White' if color == chess.WHITE else 'Black'}")
        if color == chess.BLACK:
            self.window.after(500, self.ai_move)

    def start_player_vs_player(self):
        if self.is_training and self.learn_from_human_var.get():
            messagebox.showwarning("Warning", "Stop self-play training before starting a learning game")
            return
        self.replay_mode = False
        self.board = chess.Board()
        self.human_color = None
        self.player_vs_player_mode = True
        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.last_move = None
        self.ai_thinking = False
        self.flip_board = False
        self.flip_var.set(False)
        self.reset_human_learning_game()
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Player vs Player: White to move")
    
    def watch_ai_game(self):
        self.replay_mode = False
        self.board = chess.Board()
        self.human_color = None
        self.player_vs_player_mode = False
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.last_move = None
        self.human_game_data = []
        self.human_position_counts = self.ai.new_repetition_tracker()
        self.human_game_start_time = None
        self.flip_board = False
        self.flip_var.set(False)
        self.ai_vs_ai_running = True
        self.ai_vs_ai_paused = False
        self.pause_button.config(text="Pause", state=tk.NORMAL)
        self.update_board_display()
        self.update_move_history()
        self.play_ai_vs_ai()
    
    def play_ai_vs_ai(self):
        if not self.ai_vs_ai_running:
            return
        if self.ai_vs_ai_paused:
            # Check again in 200ms without making a move
            self.window.after(200, self.play_ai_vs_ai)
            return
        if not self.ai.is_terminal_for_training(self.board):
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
        # Stop any running AI vs AI game first
        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.replay_mode = False
        self.board = chess.Board()
        self.human_color = None
        self.player_vs_player_mode = False
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.last_move = None
        self.human_game_data = []
        self.human_position_counts = self.ai.new_repetition_tracker()
        self.human_game_start_time = None
        self.flip_board = False
        self.flip_var.set(False)
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Ready")
    
    def game_over(self):
        outcome = self.board.outcome(claim_draw=True)
        if not outcome:
            msg = "Game ended"
        elif outcome.winner == chess.WHITE:
            msg = "White wins"
        elif outcome.winner == chess.BLACK:
            msg = "Black wins"
        else:
            msg = "Draw"
        msg += self.finish_human_learning_game()
        self.status_var.set(f"Game Over: {msg}")
        messagebox.showinfo("Game Over", msg)
    
    def process_queue(self):
        if self.ai.evaluation_replay_version != self._seen_evaluation_replays:
            self._seen_evaluation_replays = self.ai.evaluation_replay_version
            self.evaluation_replays = self.ai.last_evaluation_replays
            self.replay_mode = bool(self.evaluation_replays)
            self.restart_replay()
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

    def replay_replays_active(self):
        return self.replay_mode and bool(self.evaluation_replays)

    def replay_last_move(self):
        if not self.evaluation_replays or self.replay_ply <= 0:
            return None
        return chess.Move.from_uci(
            self.evaluation_replays[self.replay_game_index]['moves'][self.replay_ply - 1]
        )

    def update_replay_view(self):
        if not self.evaluation_replays:
            self.replay_info_var.set("Run an evaluation cycle to record games.")
            return
        game = self.evaluation_replays[self.replay_game_index]
        self.replay_board = chess.Board(game['positions'][self.replay_ply])
        evaluation = game['evaluations'][self.replay_ply - 1] if self.replay_ply else 0.0
        self.replay_game_var.set(
            f"Game {self.replay_game_index + 1}/{len(self.evaluation_replays)}"
        )
        self.replay_info_var.set(
            f"{game.get('result', 'Unknown')} | Ply {self.replay_ply}/{len(game['moves'])} | "
            f"Last move: {self.replay_last_move() or '-'} | Eval: {evaluation:+.3f}"
        )
        self.update_board_display()

    def select_replay_game(self, offset):
        if not self.evaluation_replays:
            return
        self.replay_mode = True
        self.replay_playing = False
        self.replay_game_index = (self.replay_game_index + offset) % len(self.evaluation_replays)
        self.replay_ply = 0
        self.update_replay_view()

    def restart_replay(self):
        if self.evaluation_replays:
            self.replay_mode = True
        self.replay_playing = False
        self.replay_game_index = min(self.replay_game_index, max(0, len(self.evaluation_replays) - 1))
        self.replay_ply = 0
        self.update_replay_view()

    def step_replay(self, direction):
        if not self.evaluation_replays:
            return
        game = self.evaluation_replays[self.replay_game_index]
        self.replay_ply = max(0, min(len(game['moves']), self.replay_ply + direction))
        self.update_replay_view()

    def toggle_replay(self):
        if not self.evaluation_replays:
            return
        self.replay_playing = not self.replay_playing
        self.replay_pause_button.config(text="Pause" if self.replay_playing else "Play")
        if self.replay_playing:
            self.advance_replay()

    def advance_replay(self):
        if not self.replay_playing or not self.evaluation_replays:
            return
        game = self.evaluation_replays[self.replay_game_index]
        if self.replay_ply >= len(game['moves']):
            self.replay_playing = False
            self.replay_pause_button.config(text="Play")
            return
        self.replay_ply += 1
        self.update_replay_view()
        self.window.after(self.replay_delay_ms, self.advance_replay)
    
    def training_callback(self, game_num, total, p_loss, v_loss, reward, game_t, train_t):
        result = "Draw" if self.ai.is_draw_reward(reward) else ("Win(W)" if reward > 0 else "Win(B)")
        text = f"Game {game_num}/{total}\n{result}\nP: {p_loss:.3f} V: {v_loss:.3f}\nGame: {game_t:.1f}s Train: {train_t:.1f}s"
        self.message_queue.put({'type': 'training_update', 'text': text})
    
    def train_worker(self, num, temp, temp_threshold):
        try:
            self.ai.train(num_games=num, temperature=temp, temp_threshold=temp_threshold, callback=self.training_callback)
            self.message_queue.put({'type': 'training_complete', 'text': f"Trained {num} games"})
        except Exception as e:
            self.message_queue.put({'type': 'training_error', 'text': str(e)})

    def reset_model(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Stop training before starting fresh")
            return
        if not messagebox.askyesno("Fresh Start", "Archive the current checkpoint and start a fresh model?"):
            return
        try:
            self.ai.reset_learning_state(archive_checkpoint=True)
            self.update_stats_display()
            self.progress_var.set("Fresh model initialized")
            self.status_var.set("Fresh model initialized")
        except Exception as e:
            messagebox.showerror("Error", str(e))
    
    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Already training")
            return
        try:
            num = int(self.num_games_var.get())
            temp = float(self.temperature_var.get())
            temp_threshold = int(self.temp_threshold_var.get())
            mcts_sims = int(self.mcts_sims_var.get())
            mcts_batch = int(self.mcts_batch_var.get())
            train_steps = int(self.train_steps_var.get())
            draw_penalty = float(self.draw_penalty_var.get())
            repeat_penalty = float(self.repeat_penalty_var.get())
            rep_draw_penalty = float(self.rep_draw_penalty_var.get())
            stockfish_path = self.stockfish_path_var.get().strip() or None
            stockfish_start = float(self.stockfish_start_var.get())
            stockfish_end = float(self.stockfish_end_var.get())
            stockfish_decay = int(self.stockfish_decay_var.get())
            resignation_threshold = float(self.resignation_threshold_var.get())
            resignation_plies = int(self.resignation_plies_var.get())
            opening_random_plies = int(self.opening_random_plies_var.get())
            evaluation_games = int(self.evaluation_games_var.get())
            evaluation_interval = int(self.evaluation_interval_var.get())
            parallel_games = int(self.parallel_games_var.get())
            if num <= 0 or temp <= 0 or temp_threshold < 0 or mcts_sims <= 0 or mcts_batch <= 0 or train_steps < 0:
                raise ValueError
            if not (0.0 <= stockfish_start <= 1.0 and 0.0 <= stockfish_end <= 1.0 and stockfish_decay > 0):
                raise ValueError
            if not (-1.0 <= resignation_threshold <= 1.0 and resignation_plies > 0 and
                    opening_random_plies >= 0 and evaluation_games >= 0 and evaluation_interval > 0 and
                    parallel_games > 0):
                raise ValueError
        except:
            messagebox.showerror("Error", "Invalid input")
            return
        self.ai.mcts_simulations = mcts_sims
        self.ai.mcts_batch_size = mcts_batch
        self.ai.train_steps_per_game = train_steps
        self.ai.draw_penalty = draw_penalty
        self.ai.repetition_penalty = repeat_penalty
        self.ai.repetition_draw_penalty = rep_draw_penalty
        if stockfish_path != self.ai.stockfish_path:
            self.ai.close_stockfish_engine()
            self.ai.stockfish_disabled_reason = ""
        self.ai.stockfish_path = stockfish_path
        self.ai.stockfish_teacher_start = stockfish_start
        self.ai.stockfish_teacher_end = stockfish_end
        self.ai.stockfish_teacher_decay_games = stockfish_decay
        self.ai.resignation_threshold = resignation_threshold
        self.ai.resignation_consecutive_plies = resignation_plies
        self.ai.opening_random_plies = opening_random_plies
        self.ai.evaluation_games = evaluation_games
        self.ai.evaluation_interval = evaluation_interval
        self.ai.parallel_games = parallel_games
        self.update_stats_display()
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.reset_model_button.config(state=tk.DISABLED)
        self.progress_var.set(f"Starting... (Temp={temp}, Sims={mcts_sims}, Switch@move {temp_threshold})")
        self.update_summary_display()
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
        self.reset_model_button.config(state=tk.NORMAL)
        self.ai.save_model()
        self.update_stats_display()
        self.status_var.set("Ready")
    
    def toggle_pause_ai_game(self):
        if not self.ai_vs_ai_running:
            return
        self.ai_vs_ai_paused = not self.ai_vs_ai_paused
        self.pause_button.config(text="Resume" if self.ai_vs_ai_paused else "Pause")
        self.status_var.set("AI vs AI paused" if self.ai_vs_ai_paused else "AI vs AI running...")
        self.update_summary_display()

    def copy_stats(self):
        text = self.stats_text.get(1.0, tk.END).strip()
        self.window.clipboard_clear()
        self.window.clipboard_append(text)
        self.status_var.set("Stats copied to clipboard")

    def copy_moves(self):
        text = self.history_text.get(1.0, tk.END).strip()
        self.window.clipboard_clear()
        self.window.clipboard_append(text)
        self.status_var.set("Move history copied to clipboard")

    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()
