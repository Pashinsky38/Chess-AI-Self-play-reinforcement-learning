"""
Chess AI with Self-Play Reinforcement Learning
IMPROVED VERSION with FIXED policy loss stability AND threefold repetition draw

Key fixes for policy loss explosion:
1. Clip log probabilities to prevent extreme negative values
2. Add maximum policy loss magnitude cap
3. Implement simple importance sampling with ratio clipping
4. Better replay buffer management with maximum age
5. Adaptive entropy coefficient
6. Enhanced diagnostics and loss monitoring
7. **NEW: Automatic draw on threefold repetition**
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

# Neural Network Architecture
class ChessNet(nn.Module):
    def __init__(self):
        super(ChessNet, self).__init__()
        # Input: 8x8x12 board representation (6 piece types x 2 colors)
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
        
        # Initialize weights
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        # Convolutional layers
        x = torch.relu(self.conv1(x))
        x = torch.relu(self.conv2(x))
        x = torch.relu(self.conv3(x))
        
        # Flatten
        x = x.view(-1, 128 * 8 * 8)
        x = torch.relu(self.fc1(x))
        x = self.dropout(x)
        
        # Policy head
        policy = self.policy_fc(x)
        
        # Value head
        value = torch.relu(self.value_fc1(x))
        value = torch.tanh(self.value_fc2(value))
        
        return policy, value


class ChessAI:
    def __init__(self, save_dir="chess_ai_models",
                 replay_capacity=10000,  # Reduced from 20000
                 batch_size=128, 
                 train_steps_per_game=4,
                 entropy_coef=0.01,  # Slightly increased
                 value_coef=1.0, 
                 clip_grad=1.0, 
                 min_buffer_size=512,
                 lr=1e-4, 
                 weight_decay=1e-5,
                 max_data_age=2000):  # NEW: max age of data in replay buffer
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        
        self.save_dir = save_dir
        self.training_stats = {
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0,
            'total_training_steps': 0  # Track training steps
        }
        self.stop_training_flag = False
        
        # Replay buffer parameters
        self.replay_capacity = replay_capacity
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.batch_size = batch_size
        self.train_steps_per_game = train_steps_per_game
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.clip_grad = clip_grad
        self.min_buffer_size = min_buffer_size
        self.max_data_age = max_data_age
        
        # NEW: Track data age for each entry
        self.data_counter = 0
        
        # NEW: Loss history for monitoring
        self.loss_history = deque(maxlen=100)
        
        os.makedirs(save_dir, exist_ok=True)
        self.load_model()
    
    def board_to_tensor(self, board):
        """Convert chess board to neural network input tensor."""
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_to_channel = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for sq in chess.SQUARES:
            p = board.piece_at(sq)
            if p:
                ch = piece_to_channel[p.piece_type]
                if p.color == chess.BLACK:
                    ch += 6
                file = chess.square_file(sq)
                rank = chess.square_rank(sq)
                tensor[ch, rank, file] = 1.0
        
        return torch.from_numpy(tensor).unsqueeze(0).to(self.device)
    
    def move_to_index(self, move):
        """Compact encoding: from*64 + to"""
        return move.from_square * 64 + move.to_square
    
    def index_to_move(self, board, idx):
        """Convert index back to legal move if possible."""
        from_sq = idx // 64
        to_sq = idx % 64
        
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

    def get_move_probabilities(self, board):
        """Get move probabilities from the neural network."""
        self.model.eval()
        with torch.no_grad():
            board_tensor = self.board_to_tensor(board)
            policy_logits, value = self.model(board_tensor)
            move_probs = torch.softmax(policy_logits, dim=1).cpu().numpy()[0]
            
            legal_moves = list(board.legal_moves)
            legal_move_probs = []
            for move in legal_moves:
                idx = self.move_to_index(move)
                legal_move_probs.append((move, float(move_probs[idx])))
            
            total_prob = sum(p for _, p in legal_move_probs)
            if total_prob > 0:
                legal_move_probs = [(m, p/total_prob) for m, p in legal_move_probs]
            else:
                prob = 1.0 / len(legal_moves)
                legal_move_probs = [(m, prob) for m in legal_moves]
            
            return legal_move_probs, float(value.item())
    
    def select_move(self, board, temperature=1.0):
        """Select a move using the neural network with exploration."""
        move_probs, _ = self.get_move_probabilities(board)
        
        if temperature == 0:
            return max(move_probs, key=lambda x: x[1])[0]
        
        moves, probs = zip(*move_probs)
        probs = np.array(probs, dtype=np.float64) ** (1.0 / temperature)
        probs = probs / probs.sum()
        return np.random.choice(moves, p=probs)
    
    def play_game(self, temperature=1.0, max_moves=100):
        """Play a single self-play game and return per-move training records."""
        board = chess.Board()
        game_data = []
        
        while not board.is_game_over() and not self.stop_training_flag:
            # **NEW: Check for threefold repetition and end game as draw**
            if board.can_claim_threefold_repetition():
                print(f"Draw by threefold repetition detected at move {len(game_data)}")
                break
            
            board_tensor = self.board_to_tensor(board).cpu()
            move = self.select_move(board, temperature)
            
            if move is None:
                break
            
            move_idx = self.move_to_index(move)
            player = board.turn
            
            # NEW: Store current policy probability for importance sampling
            with torch.no_grad():
                self.model.eval()
                board_tensor_device = board_tensor.to(self.device)
                policy_logits, _ = self.model(board_tensor_device)
                log_probs = F.log_softmax(policy_logits, dim=1)
                move_log_prob = log_probs[0, move_idx].item()
            
            game_data.append((board_tensor, move_idx, player, move_log_prob))
            board.push(move)
            
            if len(game_data) > max_moves:
                break
        
        result = board.result()
        if result == "1-0":
            reward = 1.0
            self.training_stats['white_wins'] += 1
        elif result == "0-1":
            reward = -1.0
            self.training_stats['black_wins'] += 1
        else:
            reward = 0.0  # Draw
            self.training_stats['draws'] += 1
        
        self.training_stats['games_played'] += 1
        self.training_stats['total_moves'] += len(game_data)
        
        return game_data, reward
    
    def add_game_to_buffer(self, game_data, reward):
        """Add each move from game_data to replay buffer with age tracking."""
        for board_tensor_cpu, move_idx, player, old_log_prob in game_data:
            if move_idx is None:
                continue
            
            target_value = reward if player == chess.WHITE else -reward
            
            # Store with age counter for data freshness tracking
            self.replay_buffer.append((
                board_tensor_cpu, 
                move_idx, 
                target_value, 
                old_log_prob,
                self.data_counter  # age marker
            ))
            self.data_counter += 1
    
    def clean_old_data(self):
        """Remove old data from replay buffer if it's too old."""
        if len(self.replay_buffer) < self.replay_capacity:
            return
        
        # Remove entries older than max_data_age
        current_counter = self.data_counter
        new_buffer = deque(maxlen=self.replay_capacity)
        
        for entry in self.replay_buffer:
            board_tensor, move_idx, target_value, old_log_prob, age_marker = entry
            if current_counter - age_marker < self.max_data_age:
                new_buffer.append(entry)
        
        self.replay_buffer = new_buffer
    
    def sample_batch(self):
        """Sample a batch from the replay buffer."""
        if len(self.replay_buffer) >= self.batch_size:
            batch = random.sample(self.replay_buffer, self.batch_size)
        else:
            batch = [random.choice(self.replay_buffer) for _ in range(self.batch_size)]
        
        boards, move_idxs, target_values, old_log_probs, _ = zip(*batch)
        
        boards_tensor = torch.cat(boards, dim=0).to(self.device)
        move_idxs_tensor = torch.LongTensor(move_idxs).to(self.device)
        target_values_tensor = torch.FloatTensor(target_values).to(self.device)
        old_log_probs_tensor = torch.FloatTensor(old_log_probs).to(self.device)
        
        return boards_tensor, move_idxs_tensor, target_values_tensor, old_log_probs_tensor
    
    def train_on_batch(self, boards_tensor, move_idxs_tensor, target_values_tensor, old_log_probs_tensor):
        """Train the network on a single minibatch with stabilization."""
        self.model.train()
        
        # Forward pass
        policy_logits, values = self.model(boards_tensor)
        values = values.view(-1)
        
        # Compute log probs for selected moves
        log_probs = F.log_softmax(policy_logits, dim=1)
        selected_log_probs = log_probs.gather(1, move_idxs_tensor.unsqueeze(1)).squeeze(1)
        
        # *** KEY FIX 1: Clip log probabilities to prevent extreme values ***
        selected_log_probs = torch.clamp(selected_log_probs, min=-10.0, max=0.0)
        
        # *** KEY FIX 2: Importance sampling ratio with clipping (PPO-style) ***
        # This prevents large policy updates from old data
        ratio = torch.exp(selected_log_probs - old_log_probs_tensor)
        ratio = torch.clamp(ratio, min=0.5, max=2.0)  # Clip importance sampling ratio
        
        # Compute advantages
        advantages = (target_values_tensor - values).detach()
        
        # Normalize advantages
        adv_mean = advantages.mean()
        adv_std = advantages.std(unbiased=False)
        if adv_std.item() < 1e-6:
            advantages_norm = advantages - adv_mean
        else:
            advantages_norm = (advantages - adv_mean) / (adv_std + 1e-8)
        
        # Clip advantages
        advantages_norm = torch.clamp(advantages_norm, -5.0, 5.0)  # Tighter clipping
        
        # *** KEY FIX 3: Policy loss with importance sampling and magnitude cap ***
        policy_loss_raw = -(ratio * selected_log_probs * advantages_norm).mean()
        
        # Cap the policy loss magnitude
        policy_loss = torch.clamp(policy_loss_raw, -10.0, 10.0)
        
        # Value loss
        value_loss = F.mse_loss(values, target_values_tensor)
        
        # Entropy bonus (adaptive based on training progress)
        probs = F.softmax(policy_logits, dim=1)
        entropy = -(probs * log_probs).sum(dim=1).mean()
        
        # Adaptive entropy coefficient (decreases slowly over training)
        adaptive_entropy_coef = self.entropy_coef * (1.0 + 0.1 / (1.0 + self.training_stats['total_training_steps'] / 1000.0))
        
        # Total loss
        loss = policy_loss + self.value_coef * value_loss - adaptive_entropy_coef * entropy
        
        # Safety check for non-finite loss
        if not torch.isfinite(loss):
            print("*** Non-finite loss detected - skipping update ***")
            print(f"Policy loss: {policy_loss_raw.item():.4f} (clamped to {policy_loss.item():.4f})")
            print(f"Value loss: {value_loss.item():.4f}")
            print(f"Entropy: {entropy.item():.4f}")
            print(f"Log prob range: [{selected_log_probs.min().item():.4f}, {selected_log_probs.max().item():.4f}]")
            print(f"Ratio range: [{ratio.min().item():.4f}, {ratio.max().item():.4f}]")
            return float('nan'), float('nan')
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip_grad)
        
        self.optimizer.step()
        
        # Track loss history
        self.loss_history.append(policy_loss.item())
        
        # Diagnostic: warn if policy loss magnitude is growing
        if len(self.loss_history) >= 50:
            recent_avg = sum(list(self.loss_history)[-50:]) / 50
            if abs(recent_avg) > 5.0:
                print(f"WARNING: Average policy loss magnitude is high: {recent_avg:.4f}")
        
        self.training_stats['total_training_steps'] += 1
        
        return policy_loss.item(), value_loss.item()
    
    def train(self, num_games=10, temperature=1.0, callback=None):
        """Train the AI by self-play using replay buffer + minibatches."""
        self.stop_training_flag = False
        
        for game_num in range(num_games):
            if self.stop_training_flag:
                print("Training stopped by user")
                break
            
            # Play a self-play game
            game_data, reward = self.play_game(temperature)
            
            # Add to replay buffer
            self.add_game_to_buffer(game_data, reward)
            
            # Clean old data periodically
            if game_num % 10 == 0:
                self.clean_old_data()
            
            # Training steps
            avg_policy_loss = 0.0
            avg_value_loss = 0.0
            steps_done = 0
            
            if len(self.replay_buffer) >= max(self.min_buffer_size, self.batch_size):
                for _ in range(self.train_steps_per_game):
                    boards_tensor, move_idxs_tensor, target_values_tensor, old_log_probs_tensor = self.sample_batch()
                    p_loss, v_loss = self.train_on_batch(boards_tensor, move_idxs_tensor, target_values_tensor, old_log_probs_tensor)
                    
                    if p_loss != p_loss or v_loss != v_loss:  # NaN check
                        continue
                    
                    avg_policy_loss += p_loss
                    avg_value_loss += v_loss
                    steps_done += 1
                
                if steps_done > 0:
                    avg_policy_loss /= steps_done
                    avg_value_loss /= steps_done
            else:
                avg_policy_loss, avg_value_loss = 0.0, 0.0
            
            if callback:
                callback(game_num + 1, num_games, avg_policy_loss, avg_value_loss, reward)
            
            # Save periodically
            if (game_num + 1) % 10 == 0:
                self.save_model()
    
    def stop_training(self):
        """Stop the training process"""
        self.stop_training_flag = True
    
    def save_model(self):
        """Save model and training stats."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        
        try:
            cpu_state = {k: v.cpu() for k, v in self.model.state_dict().items()}
            torch.save({
                'model_state_dict': cpu_state,
                'training_stats': self.training_stats
            }, model_path)
            
            if self.training_stats['games_played'] % 50 == 0:
                backup_path = os.path.join(self.save_dir, f"model_{timestamp}.pth")
                torch.save({
                    'model_state_dict': cpu_state,
                    'training_stats': self.training_stats
                }, backup_path)
            
            print(f"Model saved to {model_path}")
        except Exception as e:
            print(f"Error saving model: {e}")
    
    def load_model(self):
        """Load model and training stats."""
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.model.to(self.device)
                self.training_stats = checkpoint.get('training_stats', self.training_stats)
                
                # Initialize new stats if not present
                if 'total_training_steps' not in self.training_stats:
                    self.training_stats['total_training_steps'] = 0
                
                print(f"Model loaded from {model_path}")
                print(f"Training stats: {self.training_stats}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with fresh model")
        else:
            print("No saved model found. Starting fresh.")


# GUI class remains mostly the same
class ChessGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess AI - Self-Play RL (FIXED + Threefold Repetition)")
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
        """Setup the GUI components"""
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left column - Board
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10)
        
        board_container = ttk.Frame(left_frame, relief=tk.SUNKEN, borderwidth=2)
        board_container.grid(row=0, column=0, pady=10)
        
        self.board_label = ttk.Label(board_container)
        self.board_label.grid(row=0, column=0)
        self.board_label.bind("<Button-1>", self.on_board_click)
        
        # Move history
        history_frame = ttk.LabelFrame(left_frame, text="Move History", padding="5")
        history_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=10)
        
        self.history_text = scrolledtext.ScrolledText(history_frame, height=8, width=40, wrap=tk.WORD)
        self.history_text.grid(row=0, column=0)
        
        # Right column - Controls
        right_frame = ttk.Frame(main_frame)
        right_frame.grid(row=0, column=1, sticky=(tk.N, tk.W, tk.E), padx=10)
        
        # Training controls
        train_frame = ttk.LabelFrame(right_frame, text="Training Controls", padding="10")
        train_frame.grid(row=0, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Label(train_frame, text="Number of games:").grid(row=0, column=0, sticky=tk.W, pady=5)
        self.num_games_var = tk.StringVar(value="10")
        num_games_entry = ttk.Entry(train_frame, textvariable=self.num_games_var, width=15)
        num_games_entry.grid(row=0, column=1, pady=5, padx=5)
        
        ttk.Label(train_frame, text="Temperature:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.temperature_var = tk.StringVar(value="1.3")
        temp_entry = ttk.Entry(train_frame, textvariable=self.temperature_var, width=15)
        temp_entry.grid(row=1, column=1, pady=5, padx=5)
        
        button_frame = ttk.Frame(train_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        self.progress_var = tk.StringVar(value="No training in progress")
        ttk.Label(train_frame, textvariable=self.progress_var, wraplength=250).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Play controls
        play_frame = ttk.LabelFrame(right_frame, text="Play Controls", padding="10")
        play_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(play_frame, text="Play as White", command=lambda: self.start_game(chess.WHITE), width=20).grid(row=0, column=0, pady=5, padx=5)
        ttk.Button(play_frame, text="Play as Black", command=lambda: self.start_game(chess.BLACK), width=20).grid(row=1, column=0, pady=5, padx=5)
        ttk.Button(play_frame, text="AI vs AI Demo", command=self.watch_ai_game, width=20).grid(row=2, column=0, pady=5, padx=5)
        ttk.Button(play_frame, text="New Game", command=self.reset_game, width=20).grid(row=3, column=0, pady=5, padx=5)
        
        flip_cb = ttk.Checkbutton(play_frame, text="Flip board (Black at bottom)", variable=self.flip_var, command=self.on_flip_toggle)
        flip_cb.grid(row=4, column=0, pady=5)
        
        # Stats display
        stats_frame = ttk.LabelFrame(right_frame, text="Training Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=35, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0)
        
        device_info = f"Device: {self.ai.device}"
        ttk.Label(stats_frame, text=device_info, foreground="blue").grid(row=1, column=0, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - With threefold repetition draw detection")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.update_board_display()
        self.update_stats_display()
    
    def on_flip_toggle(self):
        self.flip_board = bool(self.flip_var.get())
        self.update_board_display()
    
    def board_to_image(self):
        """Convert chess board to image"""
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
        
        # Draw squares
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
        
        # Draw coordinates
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
        
        # Draw pieces
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
            print(f"Error updating board: {e}")
            self.board_label.config(text=str(self.board))
    
    def update_stats_display(self):
        stats = self.ai.training_stats
        
        win_rate_white = (stats['white_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        win_rate_black = (stats['black_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        draw_rate = (stats['draws'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        
        stats_text = f"""Games Played: {stats['games_played']}
Total Moves: {stats['total_moves']}
Training Steps: {stats.get('total_training_steps', 0)}

White Wins: {stats['white_wins']} ({win_rate_white:.1f}%)
Black Wins: {stats['black_wins']} ({win_rate_black:.1f}%)
Draws: {stats['draws']} ({draw_rate:.1f}%)

Model: {self.ai.save_dir}
        """.strip()
        
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
        if self.human_color is None or self.board.turn != self.human_color:
            return
        
        if self.board.is_game_over():
            return
        
        if self.ai_thinking:
            return
        
        offset = 20
        x, y = event.x - offset, event.y - offset
        
        if x < 0 or y < 0 or x >= self.square_size * 8 or y >= self.square_size * 8:
            return
        
        col = int(min(7, max(0, x // self.square_size)))
        row = int(min(7, max(0, y // self.square_size)))
        
        if not self.flip_board:
            file = col
            rank = 7 - row
        else:
            file = 7 - col
            rank = row
        
        square = chess.square(file, rank)
        
        if self.selected_square is None:
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                self.legal_moves_for_selected = [move for move in self.board.legal_moves if move.from_square == square]
                self.status_var.set(f"Selected {chess.SQUARE_NAMES[square]} - Click destination")
                self.update_board_display()
        else:
            move = None
            piece = self.board.piece_at(self.selected_square)
            
            if piece and piece.piece_type == chess.PAWN and (rank == 0 or rank == 7):
                promo = self.ask_promotion_piece()
                if promo is None:
                    promo = chess.QUEEN
                move = chess.Move(self.selected_square, square, promotion=promo)
            else:
                move = chess.Move(self.selected_square, square)
            
            if move in self.board.legal_moves:
                self.make_move(move)
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.update_board_display()
                
                if self.board.is_game_over() or self.board.can_claim_threefold_repetition():
                    self.game_over()
                else:
                    self.window.after(300, self.ai_move)
            else:
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.status_var.set("Illegal move! Click a piece to select it.")
                self.update_board_display()
    
    def ask_promotion_piece(self):
        dlg = tk.Toplevel(self.window)
        dlg.title("Choose Promotion")
        dlg.transient(self.window)
        dlg.grab_set()
        choice = {'piece': None}
        
        ttk.Label(dlg, text="Promote pawn to:", padding=10).grid(row=0, column=0, columnspan=4)
        
        def choose(piece_const):
            choice['piece'] = piece_const
            dlg.destroy()
        
        ttk.Button(dlg, text="Queen ♕", width=12, command=lambda: choose(chess.QUEEN)).grid(row=1, column=0, padx=5, pady=5)
        ttk.Button(dlg, text="Rook ♖", width=12, command=lambda: choose(chess.ROOK)).grid(row=1, column=1, padx=5, pady=5)
        ttk.Button(dlg, text="Bishop ♗", width=12, command=lambda: choose(chess.BISHOP)).grid(row=1, column=2, padx=5, pady=5)
        ttk.Button(dlg, text="Knight ♘", width=12, command=lambda: choose(chess.KNIGHT)).grid(row=1, column=3, padx=5, pady=5)
        
        self.window.update_idletasks()
        x = self.window.winfo_rootx() + 100
        y = self.window.winfo_rooty() + 100
        dlg.geometry(f"+{x}+{y}")
        
        dlg.wait_window()
        return choice['piece']
    
    def make_move(self, move):
        san_move = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san_move)
        self.update_move_history()
    
    def ai_move(self):
        if self.board.is_game_over() or self.board.can_claim_threefold_repetition():
            self.game_over()
            return
        
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        self.window.update()
        
        try:
            move = self.ai.select_move(self.board, temperature=0.1)
            if move is None:
                return
            
            san_move = self.board.san(move)
            self.make_move(move)
            
            self.status_var.set(f"AI played: {san_move}")
            self.update_board_display()
            
            if self.board.is_game_over() or self.board.can_claim_threefold_repetition():
                self.game_over()
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("AI Error", f"An error occurred: {e}")
        finally:
            self.ai_thinking = False
    
    def start_game(self, human_color):
        self.board = chess.Board()
        self.human_color = human_color
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.ai_thinking = False
        
        self.flip_board = (human_color == chess.BLACK)
        self.flip_var.set(self.flip_board)
        
        self.update_board_display()
        self.update_move_history()
        
        color_name = "White" if human_color == chess.WHITE else "Black"
        self.status_var.set(f"You are {color_name}. Click a piece to move.")
        
        if human_color == chess.BLACK:
            self.window.after(500, self.ai_move)
    
    def watch_ai_game(self):
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.ai_thinking = False
        
        self.flip_board = False
        self.flip_var.set(False)
        
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("AI vs AI demo started")
        
        self.play_ai_vs_ai()
    
    def play_ai_vs_ai(self):
        if not self.board.is_game_over() and not self.board.can_claim_threefold_repetition():
            try:
                move = self.ai.select_move(self.board, temperature=0.1)
                if move is None:
                    return
                
                san_move = self.board.san(move)
                self.make_move(move)
                self.update_board_display()
                self.status_var.set(f"Move: {san_move}")
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
        self.ai_thinking = False
        
        self.flip_board = False
        self.flip_var.set(False)
        
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Game reset - Choose a play mode")
    
    def game_over(self):
        result = self.board.result()
        outcome = self.board.outcome()
        
        # Check for threefold repetition
        if self.board.can_claim_threefold_repetition():
            message = "Draw by threefold repetition!"
            self.status_var.set(f"Game Over: {message} - Result: 1/2-1/2")
            messagebox.showinfo("Game Over", message)
            return
        
        if outcome is None:
            message = "Game ended"
        elif outcome.winner == chess.WHITE:
            message = "White wins!"
        elif outcome.winner == chess.BLACK:
            message = "Black wins!"
        else:
            message = "Draw!"
        
        if outcome:
            if outcome.termination == chess.Termination.CHECKMATE:
                message += " (Checkmate)"
            elif outcome.termination == chess.Termination.STALEMATE:
                message += " (Stalemate)"
            elif outcome.termination == chess.Termination.INSUFFICIENT_MATERIAL:
                message += " (Insufficient Material)"
            elif outcome.termination == chess.Termination.THREEFOLD_REPETITION:
                message += " (Threefold Repetition)"
            elif outcome.termination == chess.Termination.FIFTY_MOVES:
                message += " (Fifty Move Rule)"
        
        self.status_var.set(f"Game Over: {message} - Result: {result}")
        messagebox.showinfo("Game Over", message)
    
    def process_queue(self):
        try:
            while True:
                msg = self.message_queue.get_nowait()
                msg_type = msg.get('type')
                
                if msg_type == 'training_update':
                    self.progress_var.set(msg['text'])
                    self.update_stats_display()
                elif msg_type == 'training_complete':
                    self.stop_training()
                    messagebox.showinfo("Training Complete", msg['text'])
                elif msg_type == 'training_error':
                    self.stop_training()
                    messagebox.showerror("Training Error", msg['text'])
        except queue.Empty:
            pass
        finally:
            self.window.after(100, self.process_queue)
    
    def training_callback(self, game_num, total_games, policy_loss, value_loss, reward):
        result = "Win" if reward > 0 else ("Loss" if reward < 0 else "Draw")
        text = f"Game {game_num}/{total_games}\n{result} (R: {reward:.2f})\nP-Loss: {policy_loss:.4f}\nV-Loss: {value_loss:.4f}"
        
        self.message_queue.put({
            'type': 'training_update',
            'text': text
        })
    
    def train_worker(self, num_games, temperature):
        try:
            self.ai.train(num_games=num_games, temperature=temperature, callback=self.training_callback)
            
            self.message_queue.put({
                'type': 'training_complete',
                'text': f"Training completed! Trained on {num_games} games."
            })
        except Exception as e:
            self.message_queue.put({
                'type': 'training_error',
                'text': f"Training error: {str(e)}"
            })
    
    def start_training(self):
        if self.is_training:
            messagebox.showwarning("Already Training", "Training is already in progress!")
            return
        
        try:
            num_games = int(self.num_games_var.get())
            temperature = float(self.temperature_var.get())
            
            if num_games <= 0:
                messagebox.showerror("Invalid Input", "Number of games must be positive!")
                return
            
            if temperature <= 0:
                messagebox.showerror("Invalid Input", "Temperature must be positive!")
                return
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter valid numbers!")
            return
        
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.progress_var.set("Starting training...")
        self.status_var.set("Training in progress...")
        
        self.training_thread = threading.Thread(target=self.train_worker, args=(num_games, temperature), daemon=True)
        self.training_thread.start()
    
    def stop_training(self):
        if self.is_training:
            self.ai.stop_training()
            self.status_var.set("Stopping training...")
            self.window.after(1000, self.finish_stop_training)
        else:
            self.finish_stop_training()
    
    def finish_stop_training(self):
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ai.save_model()
        self.update_stats_display()
        self.progress_var.set("Training stopped")
        self.status_var.set("Ready")
    
    def run(self):
        self.window.mainloop()


if __name__ == "__main__":
    gui = ChessGUI()
    gui.run()