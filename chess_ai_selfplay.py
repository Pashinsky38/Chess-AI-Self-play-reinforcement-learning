"""
Chess AI with Self-Play Reinforcement Learning + Batched MCTS
IMPROVED v4 — MCTS Tree Reuse:

NEW IN THIS VERSION:
- ✅ MCTS Tree Reuse: after each move, the subtree rooted at the chosen child
  is retained and re-used as the starting point for the next search, preserving
  all visit counts and Q-values accumulated in previous simulations.
- ✅ `run_mcts_batched` accepts an optional `root_node` argument so existing
  nodes receive additional simulations rather than starting fresh.
- ✅ `select_move` maintains `self._mcts_root` and advances it after picking
  a move (including handling an opponent move via the `last_opponent_move` arg).
- ✅ `play_game` threads the reuse root across every ply of the self-play game.
- ✅ Graceful fallback: if the desired child is absent (e.g. first move, or the
  opponent played an unexpected move), a fresh root is built automatically.

RETAINED FROM v3:
- 19-channel input (piece planes + castling rights + en passant + move count)
- Squeeze-and-Excitation (SE) ResBlocks — Leela Chess Zero style channel attention
- AlphaZero-style cross-entropy policy loss (full 4096-dim MCTS visit distributions)
- Deeper policy head: 32-channel 1x1 conv
- Deeper value head: 64→256→128→1 MLP
- Label smoothing on value targets (clipped to ±0.95)
- Visit distributions stored AND correctly augmented in replay buffer
- Batched MCTS evaluation (10-50x faster than sequential)
- Mixed precision training (AMP) — 2-3x faster on RTX GPUs
- Channels-last memory format for better conv performance
- Masked softmax (no wasted compute on illegal moves)
- Dirichlet noise at root (α=0.3, ε=0.25) for exploration
- Temperature schedule (explore first 30 moves, greedy after)
- Gradient clipping (1.0) for stability
- L2 weight decay (1e-4) for regularization
- Per-move augmentation (horizontal flip, safe for castling/en-passant)
- Learning rate scheduler: warmup + cosine decay
- Draw penalty (-0.3) and inline repetition penalties
- Correct board perspective flip for Black
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


# ──────────────────────────────────────────────────────────────────────────────
# Neural Network
# ──────────────────────────────────────────────────────────────────────────────

class SEResBlock(nn.Module):
    """
    Pre-activation residual block with Squeeze-and-Excitation channel attention.

    Architecture:
      BN→ReLU→Conv3x3 → BN→ReLU→Conv3x3 → SE(scale+bias) + skip

    SE path: global avg pool → Linear(C→C//4) → ReLU → Linear(C//4→2C) →
             split into (scale, bias) → sigmoid(scale)*x + bias
    """
    def __init__(self, channels: int = 256, se_ratio: int = 4):
        super().__init__()
        se_ch = channels // se_ratio

        self.bn1   = nn.BatchNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1, bias=False)

        self.se_fc1 = nn.Linear(channels, se_ch)
        self.se_fc2 = nn.Linear(se_ch, channels * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x

        h = self.conv1(F.relu(self.bn1(x)))
        h = self.conv2(F.relu(self.bn2(h)))

        s = h.mean(dim=[2, 3])
        s = F.relu(self.se_fc1(s))
        s = self.se_fc2(s)
        scale, bias = s.chunk(2, dim=1)
        scale = torch.sigmoid(scale).unsqueeze(-1).unsqueeze(-1)
        bias  = bias.unsqueeze(-1).unsqueeze(-1)

        h = h * scale + bias
        return h + residual


class ChessNet(nn.Module):
    """
    AlphaZero-style trunk with SE residual blocks and richer input encoding.

    Input: 19-channel board tensor per position
      Ch  0-5:  current player's pieces (P,N,B,R,Q,K)
      Ch  6-11: opponent's pieces
      Ch 12:    side to move (1=current player is White, 0=Black)
      Ch 13:    current player kingside castling right
      Ch 14:    current player queenside castling right
      Ch 15:    opponent kingside castling right
      Ch 16:    opponent queenside castling right
      Ch 17:    en-passant target square (one-hot on the file)
      Ch 18:    fullmove number, normalised to [0,1] (÷100, clamped)
    """
    NUM_RES_BLOCKS = 6
    CHANNELS       = 256
    IN_CHANNELS    = 19

    def __init__(self):
        super().__init__()

        self.input_conv = nn.Conv2d(self.IN_CHANNELS, self.CHANNELS, 3, padding=1, bias=False)
        self.input_bn   = nn.BatchNorm2d(self.CHANNELS)

        self.res_blocks = nn.Sequential(
            *[SEResBlock(self.CHANNELS) for _ in range(self.NUM_RES_BLOCKS)]
        )

        self.policy_conv = nn.Conv2d(self.CHANNELS, 32, 1, bias=False)
        self.policy_bn   = nn.BatchNorm2d(32)
        self.policy_fc   = nn.Linear(32 * 8 * 8, 4096)

        self.value_conv  = nn.Conv2d(self.CHANNELS, 1, 1, bias=False)
        self.value_bn    = nn.BatchNorm2d(1)
        self.value_fc1   = nn.Linear(8 * 8, 256)
        self.value_fc2   = nn.Linear(256, 128)
        self.value_fc3   = nn.Linear(128, 1)
        self.value_drop  = nn.Dropout(0.1)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias,   0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor):
        x = F.relu(self.input_bn(self.input_conv(x)))
        x = self.res_blocks(x)

        p = F.relu(self.policy_bn(self.policy_conv(x)))
        p = p.reshape(p.size(0), -1)
        policy = self.policy_fc(p)

        v = F.relu(self.value_bn(self.value_conv(x)))
        v = v.reshape(v.size(0), -1)
        v = F.relu(self.value_fc1(v))
        v = self.value_drop(v)
        v = F.relu(self.value_fc2(v))
        value = torch.tanh(self.value_fc3(v)).squeeze(-1)

        return policy, value


# ──────────────────────────────────────────────────────────────────────────────
# Chess AI
# ──────────────────────────────────────────────────────────────────────────────

class ChessAI:

    class MCTSNode:
        __slots__ = ('board', 'parent', 'prior', 'children',
                     'visits', 'value_sum', 'virtual_loss_count')

        def __init__(self, board, parent=None, prior: float = 0.0):
            self.board               = board
            self.parent              = parent
            self.prior               = prior
            self.children: dict      = {}
            self.visits: int         = 0
            self.value_sum: float    = 0.0
            self.virtual_loss_count: int = 0

        @property
        def q_value(self) -> float:
            total = self.visits + self.virtual_loss_count
            return 0.0 if total == 0 else self.value_sum / total

    # ─── Constructor ─────────────────────────────────────────────────────────

    def __init__(self,
                 save_dir            = "chess_ai_models",
                 replay_capacity     = 30000,
                 batch_size          = 128,
                 train_steps_per_game= 16,
                 entropy_coef        = 0.005,
                 value_coef          = 1.5,
                 clip_grad           = 1.0,
                 min_buffer_size     = 200,
                 lr                  = 1e-4,
                 weight_decay        = 1e-4,
                 max_data_age        = 2000,
                 draw_penalty        = -0.3,
                 repetition_penalty  = -0.15,
                 mcts_simulations    = 128,
                 mcts_batch_size     = 8,
                 mcts_c_puct         = 1.4,
                 mcts_dirichlet_eps  = 0.25,
                 mcts_dirichlet_alpha= 0.3,
                 use_amp             = True):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model  = ChessNet().to(self.device)
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
            'tree_reuse_hits': 0,       # ← new: tracks successful reuse
            'tree_reuse_misses': 0,     # ← new: tracks cold-start fallbacks
        }
        self.stop_training_flag = False

        self.replay_capacity      = replay_capacity
        self.replay_buffer        = deque(maxlen=replay_capacity)
        self.batch_size           = batch_size
        self.train_steps_per_game = train_steps_per_game
        self.entropy_coef         = entropy_coef
        self.value_coef           = value_coef
        self.clip_grad            = clip_grad
        self.min_buffer_size      = min_buffer_size
        self.max_data_age         = max_data_age
        self.draw_penalty         = draw_penalty
        self.repetition_penalty   = repetition_penalty

        self.mcts_simulations     = mcts_simulations
        self.mcts_batch_size      = mcts_batch_size
        self.mcts_c_puct          = mcts_c_puct
        self.mcts_dirichlet_eps   = mcts_dirichlet_eps
        self.mcts_dirichlet_alpha = mcts_dirichlet_alpha

        self.data_counter = 0
        self.loss_history = deque(maxlen=100)

        # ── Tree reuse state ─────────────────────────────────────────────────
        # Holds the MCTSNode that will be used as the starting root for the
        # *next* call to run_mcts_batched.  Reset to None whenever a new game
        # starts or a cache miss occurs.
        self._mcts_root: 'ChessAI.MCTSNode | None' = None

        os.makedirs(save_dir, exist_ok=True)
        self.load_model()

    # ─── Board / move encoding ────────────────────────────────────────────────

    def board_to_tensor(self, board: chess.Board) -> torch.Tensor:
        tensor = np.zeros((19, 8, 8), dtype=np.float32)

        piece_to_ch = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN:  4, chess.KING:   5,
        }
        current = board.turn
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                ch   = piece_to_ch[p.piece_type] + (0 if p.color == current else 6)
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                if current == chess.BLACK:
                    rank = 7 - rank
                tensor[ch, rank, file] = 1.0

        tensor[12] = 1.0 if current == chess.WHITE else 0.0

        if current == chess.WHITE:
            tensor[13] = float(board.has_kingside_castling_rights(chess.WHITE))
            tensor[14] = float(board.has_queenside_castling_rights(chess.WHITE))
            tensor[15] = float(board.has_kingside_castling_rights(chess.BLACK))
            tensor[16] = float(board.has_queenside_castling_rights(chess.BLACK))
        else:
            tensor[13] = float(board.has_kingside_castling_rights(chess.BLACK))
            tensor[14] = float(board.has_queenside_castling_rights(chess.BLACK))
            tensor[15] = float(board.has_kingside_castling_rights(chess.WHITE))
            tensor[16] = float(board.has_queenside_castling_rights(chess.WHITE))

        if board.ep_square is not None:
            ep_file = chess.square_file(board.ep_square)
            ep_rank = chess.square_rank(board.ep_square)
            if current == chess.BLACK:
                ep_rank = 7 - ep_rank
            tensor[17, ep_rank, ep_file] = 1.0

        tensor[18] = min(1.0, board.fullmove_number / 100.0)

        return torch.from_numpy(tensor).unsqueeze(0)

    def move_to_index(self, move: chess.Move, flip: bool = False) -> int:
        from_sq, to_sq = move.from_square, move.to_square
        if flip:
            from_sq = chess.square(chess.square_file(from_sq), 7 - chess.square_rank(from_sq))
            to_sq   = chess.square(chess.square_file(to_sq),   7 - chess.square_rank(to_sq))
        return from_sq * 64 + to_sq

    def index_to_move(self, board: chess.Board, idx: int, flip: bool = False):
        from_sq, to_sq = idx // 64, idx % 64
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

    # ─── Augmentation ─────────────────────────────────────────────────────────

    def is_move_augmentable(self, board: chess.Board, move: chess.Move) -> bool:
        return not board.is_castling(move) and not board.is_en_passant(move)

    def _flip_visit_distribution(self, visit_dist: np.ndarray) -> np.ndarray:
        flipped = np.zeros(4096, dtype=np.float32)
        nz = np.nonzero(visit_dist)[0]
        for idx in nz:
            from_sq, to_sq = int(idx) // 64, int(idx) % 64
            new_from = (from_sq // 8) * 8 + (7 - from_sq % 8)
            new_to   = (to_sq   // 8) * 8 + (7 - to_sq   % 8)
            flipped[new_from * 64 + new_to] = visit_dist[idx]
        return flipped

    def augment(self, board_tensor: torch.Tensor, visit_dist: np.ndarray,
                can_flip: bool):
        items = [(board_tensor.clone(), visit_dist.copy())]
        if can_flip:
            items.append((
                torch.flip(board_tensor, [3]),
                self._flip_visit_distribution(visit_dist),
            ))
        return items

    # ─── Tree reuse helpers ───────────────────────────────────────────────────

    def reset_tree(self):
        """Discard the cached MCTS tree (call at the start of each new game)."""
        self._mcts_root = None

    def _advance_tree(self, move: chess.Move) -> 'ChessAI.MCTSNode | None':
        """
        Descend one level into the cached tree along `move`.

        Returns the child node (detached from the old tree) on a cache hit,
        or None on a miss.  The caller is responsible for updating
        self._mcts_root with the returned value.
        """
        if self._mcts_root is None:
            return None
        child = self._mcts_root.children.get(move)
        if child is None:
            return None
        # Detach so the rest of the old tree can be garbage-collected
        child.parent = None
        return child

    # ─── Batched network inference ─────────────────────────────────────────────

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

            mask = torch.full((4096,), -float('inf'))
            for m in legal_moves:
                idx = self.move_to_index(m, flip=flip)
                if idx < 4096:
                    mask[idx] = 0.0
            probs_t = torch.softmax(logits + mask, dim=0)

            move_probs = []
            for m in legal_moves:
                idx = self.move_to_index(m, flip=flip)
                p   = float(probs_t[idx]) if idx < 4096 else 1.0 / len(legal_moves)
                move_probs.append((m, p))

            total = sum(p for _, p in move_probs)
            if total > 0 and abs(total - 1.0) > 1e-6:
                move_probs = [(m, p / total) for m, p in move_probs]

            results.append((move_probs, value))
        return results

    def get_move_probabilities(self, board: chess.Board):
        r = self.evaluate_batch([board])
        return r[0] if r else ([], 0.0)

    # ─── Batched MCTS ──────────────────────────────────────────────────────────

    def run_mcts_batched(self,
                         root_board: chess.Board,
                         simulations=None,
                         add_dirichlet_noise: bool = False,
                         game_position_counts: dict = None,
                         root_node: 'ChessAI.MCTSNode | None' = None):
        """
        Run MCTS from `root_board`.

        Parameters
        ----------
        root_node : MCTSNode or None
            If provided, this node is used as the root instead of building a
            fresh one.  Its existing visit counts and children are preserved,
            and the requested number of *additional* simulations are run on top.
            Pass None (or omit) to start from scratch.
        """
        if simulations is None:
            simulations = self.mcts_simulations
        if game_position_counts is None:
            game_position_counts = {}

        # ── Initialise root ───────────────────────────────────────────────────
        if root_node is not None and root_node.children:
            # Reuse the existing subtree; just update the board reference so
            # the node reflects the canonical position we were asked to search.
            root = root_node
            root.board = root_board.copy()   # refresh in case of copy mismatch
            self.training_stats['tree_reuse_hits'] += 1
        else:
            # Cold start: build a new root from scratch
            root = self.MCTSNode(root_board.copy())
            move_probs, _ = self.get_move_probabilities(root.board)
            for m, prob in move_probs:
                cb = root.board.copy(); cb.push(m)
                root.children[m] = self.MCTSNode(cb, parent=root, prior=prob)
            self.training_stats['tree_reuse_misses'] += 1

        # ── Dirichlet noise at root (always applied fresh each search) ────────
        if add_dirichlet_noise and root.children:
            eps, alpha = self.mcts_dirichlet_eps, self.mcts_dirichlet_alpha
            moves = list(root.children.keys())
            noise = np.random.dirichlet([alpha] * len(moves))
            for i, m in enumerate(moves):
                root.children[m].prior = (
                    (1 - eps) * root.children[m].prior + eps * noise[i]
                )

        # ── Simulation loop ───────────────────────────────────────────────────
        num_batches = (simulations + self.mcts_batch_size - 1) // self.mcts_batch_size

        for bi in range(num_batches):
            bs = min(self.mcts_batch_size, simulations - bi * self.mcts_batch_size)
            search_paths, leaf_nodes = [], []

            for _ in range(bs):
                node, path = root, [root]
                while node.children:
                    N_total = sum(c.visits + c.virtual_loss_count
                                  for c in node.children.values()) + 1
                    best, best_m = -1e9, None
                    for m, child in node.children.items():
                        u = (self.mcts_c_puct * child.prior *
                             math.sqrt(N_total) / (1 + child.visits + child.virtual_loss_count))
                        score = child.q_value + u
                        if score > best:
                            best, best_m = score, m
                    if best_m is None:
                        break
                    node = node.children[best_m]
                    node.virtual_loss_count += 1
                    path.append(node)
                search_paths.append(path)
                leaf_nodes.append(node)

            to_eval, terminal_vals, eval_map = [], [None] * bs, []
            for i, node in enumerate(leaf_nodes):
                if node.board.is_game_over():
                    res = node.board.result()
                    if res == "1-0":
                        v = 1.0 if root_board.turn == chess.WHITE else -1.0
                    elif res == "0-1":
                        v = -1.0 if root_board.turn == chess.WHITE else 1.0
                    else:
                        v = self.draw_penalty
                    terminal_vals[i] = v
                else:
                    to_eval.append(node.board)
                    eval_map.append(i)

            eval_results = self.evaluate_batch(to_eval) if to_eval else []
            leaf_vals = list(terminal_vals)

            for ei, ni in enumerate(eval_map):
                node = leaf_nodes[ni]
                mv_probs, lv = eval_results[ei]
                prior_visits = game_position_counts.get(node.board.board_fen(), 0)
                if prior_visits >= 1:
                    lv = float(np.clip(lv + self.repetition_penalty * prior_visits, -1.0, 1.0))
                leaf_vals[ni] = lv
                for m, prob in mv_probs:
                    if m not in node.children:
                        cb = node.board.copy(); cb.push(m)
                        node.children[m] = self.MCTSNode(cb, parent=node, prior=prob)

            for path, lv in zip(search_paths, leaf_vals):
                if lv is None:
                    continue
                v = lv
                for n in reversed(path):
                    n.visits     += 1
                    n.value_sum  += v
                    if n.virtual_loss_count > 0:
                        n.virtual_loss_count -= 1
                    v = -v

        return root

    # ─── Move selection ────────────────────────────────────────────────────────

    def select_move(self,
                    board: chess.Board,
                    temperature: float = 1.0,
                    use_mcts: bool = True,
                    add_dirichlet_noise: bool = False,
                    game_position_counts: dict = None,
                    last_opponent_move: chess.Move = None):
        """
        Choose a move for the current position.

        Parameters
        ----------
        last_opponent_move : chess.Move or None
            The move the opponent just played.  When provided, the cached MCTS
            tree is descended one level along this move before searching, so
            the opponent's reply is handled as a tree-reuse step.
        """
        # ── Advance tree past the opponent's move (if any) ────────────────────
        if last_opponent_move is not None:
            reuse = self._advance_tree(last_opponent_move)
            self._mcts_root = reuse   # may be None → cold start in MCTS

        if not use_mcts:
            # Fast path: raw policy network, no MCTS
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

        # ── MCTS search (with optional tree reuse) ────────────────────────────
        root = self.run_mcts_batched(
            board,
            simulations=self.mcts_simulations,
            add_dirichlet_noise=add_dirichlet_noise,
            game_position_counts=game_position_counts,
            root_node=self._mcts_root,
        )

        if not root.children:
            self._mcts_root = None
            return None

        moves  = list(root.children.keys())
        visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)

        if temperature == 0 or temperature < 1e-8:
            move = moves[int(np.argmax(visits))]
        else:
            probs = visits ** (1.0 / temperature)
            probs /= probs.sum()
            move  = np.random.choice(moves, p=probs)

        # ── Advance the cache to the chosen child for the *next* call ─────────
        child = root.children.get(move)
        if child is not None:
            child.parent = None          # detach; let old siblings be GC'd
            self._mcts_root = child
        else:
            self._mcts_root = None

        return move

    # ─── Self-play ─────────────────────────────────────────────────────────────

    def play_game(self, temperature: float = 1.0,
                  max_moves: int = 300, temp_threshold: int = 30):
        """
        Play a self-play game with tree reuse across every ply.

        Each side's MCTS root from the previous ply is carried forward:
        - After White plays move M, the child node for M becomes the starting
          root for Black's search (which now only needs extra simulations, not
          a full cold-start).
        - After Black plays move N, the child node for N becomes the starting
          root for White's next search, and so on.
        """
        board = chess.Board()
        game_data = []
        move_count = 0
        position_counts: dict = {}

        # One cached root per side; both start cold
        reuse_root = None

        while not board.is_game_over() and not self.stop_training_flag:
            fen_key     = board.board_fen()
            visit_count = position_counts.get(fen_key, 0)
            position_counts[fen_key] = visit_count + 1

            inline_penalty = 0.0
            if visit_count >= 1:
                inline_penalty = self.repetition_penalty * visit_count
                self.training_stats['repetition_penalties_applied'] += 1

            board_tensor = self.board_to_tensor(board).cpu()

            # ── Run MCTS, reusing the subtree from two plies ago ──────────────
            root = self.run_mcts_batched(
                board,
                simulations=self.mcts_simulations,
                add_dirichlet_noise=True,
                game_position_counts=position_counts,
                root_node=reuse_root,
            )
            if not root.children:
                break

            moves  = list(root.children.keys())
            visits = np.array([root.children[m].visits for m in moves], dtype=np.float64)

            flip = (board.turn == chess.BLACK)

            # Build 4096-dim policy target from visit counts
            visit_dist = np.zeros(4096, dtype=np.float32)
            for m, v in zip(moves, visits):
                idx = self.move_to_index(m, flip=flip)
                if idx < 4096:
                    visit_dist[idx] = float(v)
            total = visit_dist.sum()
            if total > 0:
                visit_dist /= total

            # ── Sample move ───────────────────────────────────────────────────
            current_temp = temperature if move_count < temp_threshold else 0.0
            if current_temp < 1e-8:
                move = moves[int(np.argmax(visits))]
            else:
                probs = visits ** (1.0 / current_temp)
                probs /= probs.sum()
                move  = np.random.choice(moves, p=probs)

            can_flip = self.is_move_augmentable(board, move)
            player   = board.turn

            game_data.append((board_tensor, visit_dist, player, can_flip, inline_penalty))

            # ── Advance reuse root for the *opponent's* next search ───────────
            # The opponent will search from the child of the current root
            # corresponding to the move we just played.  That child has already
            # accumulated simulations from our search, giving it a warm start.
            child = root.children.get(move)
            if child is not None:
                child.parent = None      # detach from the rest of the old tree
                reuse_root = child
            else:
                reuse_root = None

            board.push(move)
            move_count += 1

            if len(game_data) > max_moves:
                break

        # ── Game outcome ──────────────────────────────────────────────────────
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
        self.training_stats['total_moves']  += len(game_data)
        return game_data, reward

    # ─── Replay buffer ─────────────────────────────────────────────────────────

    def add_game_to_buffer(self, game_data, reward: float):
        for board_tensor, visit_dist, player, can_flip, inline_penalty in game_data:
            base_value   = reward if player == chess.WHITE else -reward
            target_value = float(np.clip(base_value + inline_penalty, -0.95, 0.95))

            pairs = self.augment(board_tensor, visit_dist, can_flip)

            self.training_stats['positions_total'] += 1
            if can_flip:
                self.training_stats['positions_flipped'] += 1

            for aug_tensor, aug_dist in pairs:
                self.replay_buffer.append((
                    aug_tensor,
                    aug_dist,
                    target_value,
                    self.data_counter,
                ))
            self.data_counter += 1

    def clean_old_data(self):
        if len(self.replay_buffer) < self.replay_capacity:
            return
        cutoff = self.data_counter - self.max_data_age
        new_buf = deque(maxlen=self.replay_capacity)
        for entry in self.replay_buffer:
            if entry[3] >= cutoff:
                new_buf.append(entry)
        self.replay_buffer = new_buf

    def sample_batch(self):
        if not self.replay_buffer:
            raise ValueError("Replay buffer is empty")
        batch = random.sample(self.replay_buffer,
                              min(self.batch_size, len(self.replay_buffer)))
        if len(batch) < self.batch_size:
            batch = [random.choice(self.replay_buffer) for _ in range(self.batch_size)]

        boards, visit_dists, target_values, _ = zip(*batch)

        boards_t  = torch.cat(boards, dim=0).to(self.device)
        if self.device.type == 'cuda':
            boards_t = boards_t.to(memory_format=torch.channels_last)

        vd_t  = torch.from_numpy(np.stack(visit_dists, axis=0)).to(self.device)
        val_t = torch.FloatTensor(target_values).to(self.device)

        return boards_t, vd_t, val_t

    # ─── Training ──────────────────────────────────────────────────────────────

    def train_on_batch(self, boards_t, visit_dists_t, target_values_t):
        self.model.train()
        with torch.amp.autocast(device_type=self.device.type, enabled=self.use_amp):
            policy_logits, values = self.model(boards_t)

            log_probs = F.log_softmax(policy_logits, dim=1)

            policy_loss = -(visit_dists_t * log_probs).sum(dim=1).mean()

            value_loss  = F.mse_loss(values, target_values_t)

            probs   = torch.softmax(policy_logits, dim=1)
            entropy = -(probs * log_probs).sum(dim=1).mean()

            adaptive_ent = self.entropy_coef * (
                1.0 + 0.1 / (1.0 + self.training_stats['total_training_steps'] / 1000.0)
            )
            loss = policy_loss + self.value_coef * value_loss - adaptive_ent * entropy

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

        self.loss_history.append(policy_loss.item())
        self.training_stats['total_training_steps'] += 1
        try:
            self.scheduler.step()
        except Exception:
            pass

        return policy_loss.item(), value_loss.item()

    def train(self, num_games: int = 10, temperature: float = 1.0,
              temp_threshold: int = 30, callback=None):
        self.stop_training_flag = False

        for game_num in range(num_games):
            if self.stop_training_flag:
                break

            t0 = time.time()
            game_data, reward = self.play_game(temperature, temp_threshold=temp_threshold)
            game_time = time.time() - t0

            self.add_game_to_buffer(game_data, reward)
            if game_num % 10 == 0:
                self.clean_old_data()

            p_loss_avg = v_loss_avg = 0.0
            steps = 0
            t1 = time.time()

            if len(self.replay_buffer) >= max(self.min_buffer_size, self.batch_size):
                for _ in range(self.train_steps_per_game):
                    try:
                        b, vd, tv = self.sample_batch()
                    except ValueError:
                        break
                    pl, vl = self.train_on_batch(b, vd, tv)
                    if math.isnan(pl):
                        continue
                    p_loss_avg += pl
                    v_loss_avg += vl
                    steps += 1
                if steps:
                    p_loss_avg /= steps
                    v_loss_avg /= steps

            train_time = time.time() - t1

            if callback:
                callback(game_num + 1, num_games, p_loss_avg, v_loss_avg,
                         reward, game_time, train_time)
            if (game_num + 1) % 10 == 0:
                self.save_model()

    def stop_training(self):
        self.stop_training_flag = True

    # ─── Save / Load ───────────────────────────────────────────────────────────

    def save_model(self):
        path = os.path.join(self.save_dir, "model_latest.pth")
        try:
            cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            d = {
                'model_state_dict':     cpu_state,
                'training_stats':       self.training_stats,
                'optimizer_state_dict': self.optimizer.state_dict(),
                'scheduler_state_dict': self.scheduler.state_dict(),
                'arch_version':         3,
            }
            if self.scaler:
                d['scaler_state_dict'] = self.scaler.state_dict()
            torch.save(d, path)
            if self.training_stats['games_played'] % 50 == 0:
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                torch.save(d, os.path.join(self.save_dir, f"model_{ts}.pth"))
        except Exception as e:
            print(f"Error saving model: {e}")

    def load_model(self):
        path = os.path.join(self.save_dir, "model_latest.pth")
        if not os.path.exists(path):
            print("No saved model found. Starting fresh.")
            return
        try:
            ckpt = torch.load(path, map_location='cpu')
            if ckpt.get('arch_version', 1) < 3:
                print("Saved model is from an older architecture. Starting fresh.")
                return
            self.model.load_state_dict(ckpt['model_state_dict'])
            self.model.to(self.device)
            if self.device.type == 'cuda':
                self.model = self.model.to(memory_format=torch.channels_last)
            self.training_stats = ckpt.get('training_stats', self.training_stats)
            for k in ['positions_flipped', 'positions_total',
                      'total_training_steps', 'repetition_penalties_applied',
                      'tree_reuse_hits', 'tree_reuse_misses']:
                self.training_stats.setdefault(k, 0)
            if 'optimizer_state_dict' in ckpt:
                self.optimizer.load_state_dict(ckpt['optimizer_state_dict'])
            if 'scheduler_state_dict' in ckpt:
                self.scheduler.load_state_dict(ckpt['scheduler_state_dict'])
            if self.scaler and 'scaler_state_dict' in ckpt:
                self.scaler.load_state_dict(ckpt['scaler_state_dict'])
            print(f"Model loaded from {path}")
            print(f"Stats: {self.training_stats}")
        except Exception as e:
            print(f"Error loading model: {e}. Starting fresh.")


# ──────────────────────────────────────────────────────────────────────────────
# GUI
# ──────────────────────────────────────────────────────────────────────────────

class ChessGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess AI v4 — Tree Reuse + SE-ResNet + AlphaZero Training")
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
        self.ai_vs_ai_running = False
        self.ai_vs_ai_paused = False

        # Track the last move played (for tree reuse in human vs AI games)
        self._last_human_move: chess.Move | None = None

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

        # Training controls
        train_frame = ttk.LabelFrame(right_frame, text="Training Controls", padding="10")
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Label(train_frame, text="Number of games:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_games_var = tk.StringVar(value="10")
        ttk.Entry(train_frame, textvariable=self.num_games_var, width=15).grid(row=0, column=1, pady=5, padx=5)

        ttk.Label(train_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="1.0")
        ttk.Entry(train_frame, textvariable=self.temperature_var, width=15).grid(row=1, column=1, pady=5, padx=5)

        ttk.Label(train_frame, text="Temp threshold:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.temp_threshold_var = tk.StringVar(value="30")
        ttk.Entry(train_frame, textvariable=self.temp_threshold_var, width=15).grid(row=2, column=1, pady=5, padx=5)
        ttk.Label(train_frame, text="(explore N moves then greedy)",
                  font=('Arial', 8), foreground='gray').grid(row=2, column=2, sticky=tk.W, padx=5)

        bf = ttk.Frame(train_frame)
        bf.grid(row=3, column=0, columnspan=3, pady=10)
        self.train_button = ttk.Button(bf, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5)
        self.stop_button = ttk.Button(bf, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)

        self.progress_var = tk.StringVar(value="No training in progress")
        ttk.Label(train_frame, textvariable=self.progress_var, wraplength=250).grid(
            row=4, column=0, columnspan=3, pady=5)

        # Play controls
        play_frame = ttk.LabelFrame(right_frame, text="Play Controls", padding="10")
        play_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)

        ttk.Button(play_frame, text="Play as White",  command=lambda: self.start_game(chess.WHITE), width=20).grid(row=0, column=0, pady=5)
        ttk.Button(play_frame, text="Play as Black",  command=lambda: self.start_game(chess.BLACK), width=20).grid(row=1, column=0, pady=5)
        ttk.Button(play_frame, text="AI vs AI Demo",  command=self.watch_ai_game, width=20).grid(row=2, column=0, pady=5)
        self.pause_button = ttk.Button(play_frame, text="Pause", command=self.toggle_pause_ai_game, width=20, state=tk.DISABLED)
        self.pause_button.grid(row=3, column=0, pady=5)
        ttk.Button(play_frame, text="New Game", command=self.reset_game, width=20).grid(row=4, column=0, pady=5)
        ttk.Checkbutton(play_frame, text="Flip board", variable=self.flip_var, command=self.on_flip_toggle).grid(row=5, column=0, pady=5)

        # Stats
        stats_frame = ttk.LabelFrame(right_frame, text="Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        self.stats_text = tk.Text(stats_frame, height=20, width=35, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0)
        ttk.Button(stats_frame, text="Copy Stats", command=self.copy_stats, width=15).grid(row=1, column=0, pady=2)

        amp_status = "AMP: ON" if self.ai.use_amp else "AMP: OFF"
        param_count = sum(p.numel() for p in self.ai.model.parameters()) / 1e6
        ttk.Label(stats_frame, text=f"Device: {self.ai.device}",        foreground="blue").grid(row=2, column=0, pady=1)
        ttk.Label(stats_frame, text=amp_status,                          foreground="green").grid(row=3, column=0, pady=1)
        ttk.Label(stats_frame, text=f"Model: {param_count:.1f}M params", foreground="purple").grid(row=4, column=0, pady=1)
        ttk.Label(stats_frame, text="✅ SE-ResNet  |  AlphaZero Loss  |  Tree Reuse",
                  foreground="red", font=('Arial', 9, 'bold')).grid(row=5, column=0, pady=1)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W).grid(
            row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)

        self.update_board_display()
        self.update_stats_display()

    def on_flip_toggle(self):
        self.flip_board = bool(self.flip_var.get())
        self.update_board_display()

    def board_to_image(self):
        board_size = self.square_size * 8
        image = Image.new('RGB', (board_size + 40, board_size + 40), 'white')
        draw  = ImageDraw.Draw(image)

        light_sq     = (240, 217, 181)
        dark_sq      = (181, 136,  99)
        selected_col = (255, 255, 100)
        legal_col    = (144, 238, 144)

        try:
            piece_font = ImageFont.truetype("seguisym.ttf", int(self.square_size * 0.7))
            coord_font = ImageFont.truetype("arial.ttf", 12)
        except Exception:
            try:
                piece_font = ImageFont.truetype("Arial.ttf", int(self.square_size * 0.7))
                coord_font = piece_font
            except Exception:
                piece_font = coord_font = ImageFont.load_default()

        off = 20
        legal_targets = {m.to_square for m in self.legal_moves_for_selected}

        for rank in range(8):
            for file in range(8):
                x1 = (file if not self.flip_board else 7 - file) * self.square_size + off
                y1 = ((7 - rank) if not self.flip_board else rank) * self.square_size + off
                x2, y2 = x1 + self.square_size, y1 + self.square_size
                sq = chess.square(file, rank)
                if sq == self.selected_square:
                    color = selected_col
                elif sq in legal_targets:
                    color = legal_col
                elif (rank + file) % 2 == 0:
                    color = light_sq
                else:
                    color = dark_sq
                draw.rectangle([x1, y1, x2, y2], fill=color, outline='gray')

        files = ['a','b','c','d','e','f','g','h']
        ranks = ['1','2','3','4','5','6','7','8']
        if self.flip_board:
            files, ranks = list(reversed(files)), list(reversed(ranks))

        for i, ch in enumerate(files):
            x = i * self.square_size + self.square_size // 2 + off
            draw.text((x, 5),                   ch, fill='black', font=coord_font, anchor='mm')
            draw.text((x, board_size + off + 15), ch, fill='black', font=coord_font, anchor='mm')
        for i, ch in enumerate(ranks):
            y = ((7 - i) if not self.flip_board else i) * self.square_size + self.square_size // 2 + off
            draw.text((5,                   y), ch, fill='black', font=coord_font, anchor='mm')
            draw.text((board_size + off + 15, y), ch, fill='black', font=coord_font, anchor='mm')

        symbols = {'P':'♙','N':'♘','B':'♗','R':'♖','Q':'♕','K':'♔',
                   'p':'♟','n':'♞','b':'♝','r':'♜','q':'♛','k':'♚'}
        for rank in range(8):
            for file in range(8):
                sq = chess.square(file, rank)
                p  = self.board.piece_at(sq)
                if not p:
                    continue
                ch = symbols.get(p.symbol(), p.symbol())
                x = (file if not self.flip_board else 7 - file) * self.square_size + self.square_size // 2 + off
                y = ((7 - rank) if not self.flip_board else rank) * self.square_size + self.square_size // 2 + off
                pc = 'white' if p.color == chess.WHITE else 'black'
                oc = 'black' if p.color == chess.WHITE else 'white'
                for dx, dy in [(-1,-1),(-1,1),(1,-1),(1,1)]:
                    draw.text((x+dx, y+dy), ch, fill=oc, font=piece_font, anchor='mm')
                draw.text((x, y), ch, fill=pc, font=piece_font, anchor='mm')

        return ImageTk.PhotoImage(image)

    def update_board_display(self):
        try:
            photo = self.board_to_image()
            self.board_label.config(image=photo)
            self.board_label.image = photo
        except Exception as e:
            self.board_label.config(text=str(self.board))

    def update_stats_display(self):
        s = self.ai.training_stats
        gp = max(s['games_played'], 1)
        ww = s['white_wins'] / gp * 100
        bw = s['black_wins'] / gp * 100
        dr = s['draws']      / gp * 100
        pt = max(s['positions_total'], 1)
        fr = s['positions_flipped'] / pt * 100
        lr = self.ai.optimizer.param_groups[0]['lr']
        hits   = s.get('tree_reuse_hits', 0)
        misses = s.get('tree_reuse_misses', 0)
        total_searches = max(hits + misses, 1)
        hit_rate = hits / total_searches * 100

        txt = (f"Games:  {s['games_played']}\n"
               f"Moves:  {s['total_moves']}\n"
               f"Steps:  {s['total_training_steps']}\n\n"
               f"White:  {s['white_wins']} ({ww:.1f}%)\n"
               f"Black:  {s['black_wins']} ({bw:.1f}%)\n"
               f"Draws:  {s['draws']} ({dr:.1f}%)\n\n"
               f"Buffer: {len(self.ai.replay_buffer)}\n"
               f"Augm:   {fr:.1f}% flipped\n"
               f"RepPen: {s['repetition_penalties_applied']}\n"
               f"LR:     {lr:.2e}\n\n"
               f"── Tree Reuse ──\n"
               f"Hits:   {hits} ({hit_rate:.1f}%)\n"
               f"Misses: {misses}\n\n"
               f"DrawP:  {self.ai.draw_penalty}\n"
               f"RepP:   {self.ai.repetition_penalty}\n"
               f"Dir:    {self.ai.save_dir}")
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, txt)

    def update_move_history(self):
        self.history_text.delete(1.0, tk.END)
        if not self.move_history:
            self.history_text.insert(1.0, "No moves yet")
            return
        text = ""
        for i, m in enumerate(self.move_history):
            text += (f"{i//2+1}. {m} " if i % 2 == 0 else f"{m}\n")
        self.history_text.insert(1.0, text)
        self.history_text.see(tk.END)

    def on_board_click(self, event):
        if (self.human_color is None or self.board.turn != self.human_color
                or self.board.is_game_over() or self.ai_thinking):
            return
        off = 20
        x, y = event.x - off, event.y - off
        if not (0 <= x < self.square_size*8 and 0 <= y < self.square_size*8):
            return
        col = int(min(7, max(0, x // self.square_size)))
        row = int(min(7, max(0, y // self.square_size)))
        file, rank = ((col, 7-row) if not self.flip_board else (7-col, row))
        sq = chess.square(file, rank)

        if self.selected_square is None:
            p = self.board.piece_at(sq)
            if p and p.color == self.human_color:
                self.selected_square = sq
                self.legal_moves_for_selected = [m for m in self.board.legal_moves if m.from_square == sq]
                self.status_var.set(f"Selected {chess.SQUARE_NAMES[sq]}")
                self.update_board_display()
        else:
            p = self.board.piece_at(self.selected_square)
            promo = None
            if p and p.piece_type == chess.PAWN and rank in (0, 7):
                promo = self.ask_promotion_piece() or chess.QUEEN
            move = chess.Move(self.selected_square, sq, promotion=promo)
            if move in self.board.legal_moves:
                self._last_human_move = move
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
        for col, (sym, pc) in enumerate([('♕', chess.QUEEN), ('♖', chess.ROOK),
                                          ('♗', chess.BISHOP), ('♘', chess.KNIGHT)]):
            ttk.Button(dlg, text=sym, width=8,
                       command=lambda p=pc: [choice.update({'piece': p}), dlg.destroy()]).grid(
                row=1, column=col, padx=5, pady=5)
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
            # Pass the human's last move so the AI can reuse the subtree
            move = self.ai.select_move(
                self.board,
                temperature=0.0,
                use_mcts=True,
                last_opponent_move=self._last_human_move,
            )
            self._last_human_move = None   # consumed
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
        self._last_human_move = None
        self.ai.reset_tree()        # ← clear any stale cached tree
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
        self.ai_vs_ai_running = True
        self.ai_vs_ai_paused  = False
        self.pause_button.config(text="Pause", state=tk.NORMAL)
        self.ai.reset_tree()        # ← fresh tree for the demo game
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
                # In the demo, _last_human_move is None — the AI manages its
                # own tree internally via self.ai._mcts_root across turns.
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
        self.ai_vs_ai_running = self.ai_vs_ai_paused = False
        self.pause_button.config(text="Pause", state=tk.DISABLED)
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.flip_board = False
        self.flip_var.set(False)
        self._last_human_move = None
        self.ai.reset_tree()
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Ready")

    def game_over(self):
        outcome = self.board.outcome()
        if not outcome:
            msg = "Game ended"
        elif outcome.winner == chess.WHITE:
            msg = "White wins!"
        elif outcome.winner == chess.BLACK:
            msg = "Black wins!"
        else:
            msg = "Draw"
        self.ai.reset_tree()   # stale tree is no longer useful
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
                  else ("Win(W)" if reward > 0 else "Win(B)"))
        text = (f"Game {game_num}/{total}\n{result}\n"
                f"P: {p_loss:.3f}  V: {v_loss:.3f}\n"
                f"Game: {game_t:.1f}s  Train: {train_t:.1f}s")
        self.message_queue.put({'type': 'training_update', 'text': text})

    def train_worker(self, num, temp, temp_threshold):
        try:
            self.ai.train(num_games=num, temperature=temp,
                          temp_threshold=temp_threshold,
                          callback=self.training_callback)
            self.message_queue.put({'type': 'training_complete',
                                    'text': f"Trained {num} games successfully."})
        except Exception as e:
            self.message_queue.put({'type': 'training_error', 'text': str(e)})

    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Warning", "Already training"); return
        try:
            num       = int(self.num_games_var.get())
            temp      = float(self.temperature_var.get())
            temp_thr  = int(self.temp_threshold_var.get())
            if num <= 0 or temp <= 0 or temp_thr < 0:
                raise ValueError
        except Exception:
            messagebox.showerror("Error", "Invalid input"); return
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set(f"Starting... (T={temp}, switch@move {temp_thr})")
        self.training_thread = threading.Thread(
            target=self.train_worker, args=(num, temp, temp_thr), daemon=True)
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
        self.status_var.set("Paused" if self.ai_vs_ai_paused else "AI vs AI running...")

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