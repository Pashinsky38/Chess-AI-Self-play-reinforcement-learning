"""
Chess AI with Self-Play Reinforcement Learning
IMPROVED VERSION with enhanced GUI and user experience
Windows-compatible version - no cairosvg required!
"""

import chess
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from PIL import Image, ImageDraw, ImageFont, ImageTk
import threading
import queue

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
    def __init__(self, save_dir="chess_ai_models"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = ChessNet().to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.save_dir = save_dir
        self.training_stats = {
            'games_played': 0,
            'total_moves': 0,
            'white_wins': 0,
            'black_wins': 0,
            'draws': 0
        }
        self.stop_training_flag = False
        
        # Create save directory if it doesn't exist
        os.makedirs(save_dir, exist_ok=True)
        
        # Try to load existing model
        self.load_model()
    
    def board_to_tensor(self, board):
        """Convert chess board to neural network input tensor"""
        # 12 channels: 6 piece types * 2 colors
        tensor = np.zeros((12, 8, 8), dtype=np.float32)
        
        piece_to_channel = {
            chess.PAWN: 0, chess.KNIGHT: 1, chess.BISHOP: 2,
            chess.ROOK: 3, chess.QUEEN: 4, chess.KING: 5
        }
        
        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                channel = piece_to_channel[piece.piece_type]
                if piece.color == chess.BLACK:
                    channel += 6
                row, col = divmod(square, 8)
                tensor[channel, row, col] = 1.0
        
        return torch.FloatTensor(tensor).unsqueeze(0).to(self.device)
    
    def get_move_probabilities(self, board):
        """Get move probabilities from the neural network"""
        self.model.eval()
        with torch.no_grad():
            board_tensor = self.board_to_tensor(board)
            policy, value = self.model(board_tensor)
            
            # Mask illegal moves
            legal_moves = list(board.legal_moves)
            move_probs = torch.softmax(policy, dim=1).cpu().numpy()[0]
            
            # Convert moves to indices
            legal_move_probs = []
            for move in legal_moves:
                from_sq = move.from_square
                to_sq = move.to_square
                move_idx = from_sq * 64 + to_sq
                legal_move_probs.append((move, move_probs[move_idx]))
            
            # Normalize probabilities
            total_prob = sum(prob for _, prob in legal_move_probs)
            if total_prob > 0:
                legal_move_probs = [(move, prob/total_prob) for move, prob in legal_move_probs]
            else:
                # If all probs are zero, use uniform distribution
                prob = 1.0 / len(legal_moves)
                legal_move_probs = [(move, prob) for move in legal_moves]
            
            return legal_move_probs, value.item()
    
    def select_move(self, board, temperature=1.0):
        """Select a move using the neural network with exploration"""
        move_probs, _ = self.get_move_probabilities(board)
        
        if temperature == 0:
            # Greedy selection
            return max(move_probs, key=lambda x: x[1])[0]
        
        # Sample from probability distribution with temperature
        moves, probs = zip(*move_probs)
        probs = np.array(probs) ** (1.0 / temperature)
        probs = probs / probs.sum()
        
        return np.random.choice(moves, p=probs)
    
    def play_game(self, temperature=1.0):
        """Play a single self-play game"""
        board = chess.Board()
        game_data = []
        
        while not board.is_game_over() and not self.stop_training_flag:
            # Store board state
            board_tensor = self.board_to_tensor(board)
            
            # Select move
            move = self.select_move(board, temperature)
            
            # Store state and move for training
            game_data.append({
                'board': board_tensor.cpu(),
                'move': move,
                'player': board.turn
            })
            
            board.push(move)
            
            # Limit game length to prevent infinite games
            if len(game_data) > 200:
                break
        
        # Determine game result
        result = board.result()
        if result == "1-0":
            reward = 1.0
            self.training_stats['white_wins'] += 1
        elif result == "0-1":
            reward = -1.0
            self.training_stats['black_wins'] += 1
        else:
            reward = 0.0
            self.training_stats['draws'] += 1
        
        self.training_stats['games_played'] += 1
        self.training_stats['total_moves'] += len(game_data)
        
        return game_data, reward
    
    def train_on_game(self, game_data, reward):
        """Train the network on a completed game"""
        if len(game_data) == 0:
            return 0, 0
            
        self.model.train()
        
        total_policy_loss = 0
        total_value_loss = 0
        
        for i, data in enumerate(game_data):
            board_tensor = data['board'].to(self.device)
            move = data['move']
            player = data['player']
            
            # Adjust reward based on player
            target_value = reward if player == chess.WHITE else -reward
            
            # Forward pass
            policy, value = self.model(board_tensor)
            
            # Value loss
            value_loss = (value - target_value) ** 2
            
            # Policy loss (encourage winning moves)
            move_idx = move.from_square * 64 + move.to_square
            policy_loss = -torch.log_softmax(policy, dim=1)[0, move_idx] * target_value
            
            # Combined loss
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()
        
        return total_policy_loss / len(game_data), total_value_loss / len(game_data)
    
    def train(self, num_games=10, temperature=1.0, callback=None):
        """Train the AI by self-play"""
        self.stop_training_flag = False
        
        for game_num in range(num_games):
            if self.stop_training_flag:
                print("Training stopped by user")
                break
                
            # Play a game
            game_data, reward = self.play_game(temperature)
            
            # Train on the game
            policy_loss, value_loss = self.train_on_game(game_data, reward)
            
            if callback:
                callback(game_num + 1, num_games, policy_loss, value_loss, reward)
            
            # Save model periodically
            if (game_num + 1) % 10 == 0:
                self.save_model()
    
    def stop_training(self):
        """Stop the training process"""
        self.stop_training_flag = True
    
    def save_model(self):
        """Save model and training stats"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save model weights
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'training_stats': self.training_stats
        }, model_path)
        
        # Also save a timestamped backup every 50 games
        if self.training_stats['games_played'] % 50 == 0:
            backup_path = os.path.join(self.save_dir, f"model_{timestamp}.pth")
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'training_stats': self.training_stats
            }, backup_path)
        
        print(f"Model saved to {model_path}")
    
    def load_model(self):
        """Load model and training stats"""
        model_path = os.path.join(self.save_dir, "model_latest.pth")
        if os.path.exists(model_path):
            try:
                checkpoint = torch.load(model_path, map_location=self.device)
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                self.training_stats = checkpoint['training_stats']
                print(f"Model loaded from {model_path}")
                print(f"Training stats: {self.training_stats}")
            except Exception as e:
                print(f"Error loading model: {e}")
                print("Starting with fresh model")
        else:
            print("No saved model found. Starting fresh.")


class ChessGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess AI - Self-Play Reinforcement Learning (IMPROVED)")
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
        
        # Message queue for thread-safe GUI updates
        self.message_queue = queue.Queue()
        
        self.setup_gui()
        self.process_queue()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main container with two columns
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Left column - Board
        left_frame = ttk.Frame(main_frame)
        left_frame.grid(row=0, column=0, padx=10)
        
        # Board display
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
        
        # Right column - Controls and Stats
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
        self.temperature_var = tk.StringVar(value="1.0")
        temp_entry = ttk.Entry(train_frame, textvariable=self.temperature_var, width=15)
        temp_entry.grid(row=1, column=1, pady=5, padx=5)
        
        button_frame = ttk.Frame(train_frame)
        button_frame.grid(row=2, column=0, columnspan=2, pady=10)
        
        self.train_button = ttk.Button(button_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=0, padx=5)
        
        self.stop_button = ttk.Button(button_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=1, padx=5)
        
        # Training progress
        self.progress_var = tk.StringVar(value="No training in progress")
        ttk.Label(train_frame, textvariable=self.progress_var, wraplength=250).grid(row=3, column=0, columnspan=2, pady=5)
        
        # Play controls
        play_frame = ttk.LabelFrame(right_frame, text="Play Controls", padding="10")
        play_frame.grid(row=1, column=0, sticky=(tk.W, tk.E), pady=5)
        
        ttk.Button(play_frame, text="Play as White", command=lambda: self.start_game(chess.WHITE), width=20).grid(row=0, column=0, pady=5, padx=5)
        ttk.Button(play_frame, text="Play as Black", command=lambda: self.start_game(chess.BLACK), width=20).grid(row=1, column=0, pady=5, padx=5)
        ttk.Button(play_frame, text="AI vs AI Demo", command=self.watch_ai_game, width=20).grid(row=2, column=0, pady=5, padx=5)
        ttk.Button(play_frame, text="New Game", command=self.reset_game, width=20).grid(row=3, column=0, pady=5, padx=5)
        
        # Stats display
        stats_frame = ttk.LabelFrame(right_frame, text="Training Statistics", padding="10")
        stats_frame.grid(row=2, column=0, sticky=(tk.W, tk.E), pady=5)
        
        self.stats_text = tk.Text(stats_frame, height=10, width=35, wrap=tk.WORD)
        self.stats_text.grid(row=0, column=0)
        
        # Device info
        device_info = f"Device: {self.ai.device}"
        ttk.Label(stats_frame, text=device_info, foreground="blue").grid(row=1, column=0, pady=5)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready - Click 'Play as White/Black' or 'Start Training'")
        status_label = ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN, anchor=tk.W)
        status_label.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        self.update_board_display()
        self.update_stats_display()
    
    def board_to_image(self):
        """Convert chess board to image using PIL with improved rendering"""
        board_size = self.square_size * 8
        
        # Create image with extra space for coordinates
        image = Image.new('RGB', (board_size + 40, board_size + 40), 'white')
        draw = ImageDraw.Draw(image)
        
        # Colors
        light_square = (240, 217, 181)
        dark_square = (181, 136, 99)
        selected_color = (255, 255, 100)
        legal_move_color = (144, 238, 144)
        
        # Try to load a font
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
        
        # Offset for board (to make room for coordinates)
        offset = 20
        
        # Draw squares
        for rank in range(8):
            for file in range(8):
                x1 = file * self.square_size + offset
                y1 = (7 - rank) * self.square_size + offset
                x2 = x1 + self.square_size
                y2 = y1 + self.square_size
                
                square = chess.square(file, rank)
                
                # Determine square color
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
        
        for i, file_char in enumerate(files):
            x = i * self.square_size + self.square_size // 2 + offset
            draw.text((x, 5), file_char, fill='black', font=coord_font, anchor='mm')
            draw.text((x, board_size + offset + 15), file_char, fill='black', font=coord_font, anchor='mm')
        
        for i, rank_char in enumerate(ranks):
            y = (7 - i) * self.square_size + self.square_size // 2 + offset
            draw.text((5, y), rank_char, fill='black', font=coord_font, anchor='mm')
            draw.text((board_size + offset + 15, y), rank_char, fill='black', font=coord_font, anchor='mm')
        
        # Unicode chess pieces
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
                    
                    x = file * self.square_size + self.square_size // 2 + offset
                    y = (7 - rank) * self.square_size + self.square_size // 2 + offset
                    
                    # Piece color
                    piece_color = 'white' if piece.color == chess.WHITE else 'black'
                    outline_color = 'black' if piece.color == chess.WHITE else 'white'
                    
                    # Draw piece with outline for better visibility
                    for dx, dy in [(-1, -1), (-1, 1), (1, -1), (1, 1)]:
                        draw.text((x + dx, y + dy), piece_char, fill=outline_color, 
                                 font=piece_font, anchor='mm')
                    draw.text((x, y), piece_char, fill=piece_color, 
                             font=piece_font, anchor='mm')
        
        return ImageTk.PhotoImage(image)
    
    def update_board_display(self):
        """Update the board display"""
        try:
            photo = self.board_to_image()
            self.board_label.config(image=photo)
            self.board_label.image = photo  # Keep a reference
        except Exception as e:
            print(f"Error updating board: {e}")
            # Fallback to text representation
            self.board_label.config(text=str(self.board))
    
    def update_stats_display(self):
        """Update statistics display"""
        stats = self.ai.training_stats
        
        win_rate_white = (stats['white_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        win_rate_black = (stats['black_wins'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        draw_rate = (stats['draws'] / stats['games_played'] * 100) if stats['games_played'] > 0 else 0
        
        stats_text = f"""Games Played: {stats['games_played']}
Total Moves: {stats['total_moves']}

White Wins: {stats['white_wins']} ({win_rate_white:.1f}%)
Black Wins: {stats['black_wins']} ({win_rate_black:.1f}%)
Draws: {stats['draws']} ({draw_rate:.1f}%)

Model Location:
{self.ai.save_dir}
        """.strip()
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def update_move_history(self):
        """Update move history display"""
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
        """Handle board click for human moves"""
        if self.human_color is None or self.board.turn != self.human_color:
            return
        
        if self.board.is_game_over():
            return
        
        if self.ai_thinking:
            return
        
        # Calculate which square was clicked (accounting for offset)
        offset = 20
        x, y = event.x - offset, event.y - offset
        
        # Check if click is within board bounds
        if x < 0 or y < 0 or x >= self.square_size * 8 or y >= self.square_size * 8:
            return
        
        file = min(7, max(0, x // self.square_size))
        rank = min(7, max(0, 7 - (y // self.square_size)))
        square = chess.square(file, rank)
        
        if self.selected_square is None:
            # Select piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                # Get legal moves for this piece
                self.legal_moves_for_selected = [
                    move for move in self.board.legal_moves 
                    if move.from_square == square
                ]
                self.status_var.set(f"Selected {chess.SQUARE_NAMES[square]} - Click destination square")
                self.update_board_display()
        else:
            # Try to make move
            move = None
            
            # Check for promotion
            piece = self.board.piece_at(self.selected_square)
            if piece and piece.piece_type == chess.PAWN and (rank == 0 or rank == 7):
                # Pawn promotion - default to queen
                move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
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
                    # AI's turn
                    self.window.after(300, self.ai_move)
            else:
                # Invalid move - deselect
                self.selected_square = None
                self.legal_moves_for_selected = []
                self.status_var.set("Illegal move! Click a piece to select it.")
                self.update_board_display()
    
    def make_move(self, move):
        """Make a move on the board and update history"""
        san_move = self.board.san(move)
        self.board.push(move)
        self.move_history.append(san_move)
        self.update_move_history()
    
    def ai_move(self):
        """Make AI move"""
        if self.board.is_game_over():
            self.game_over()
            return
        
        self.ai_thinking = True
        self.status_var.set("AI thinking...")
        self.window.update()
        
        try:
            move = self.ai.select_move(self.board, temperature=0.1)
            san_move = self.board.san(move)
            self.make_move(move)
            
            self.status_var.set(f"AI played: {san_move}")
            self.update_board_display()
            
            if self.board.is_game_over():
                self.game_over()
        except Exception as e:
            self.status_var.set(f"Error: {e}")
            messagebox.showerror("AI Error", f"An error occurred: {e}")
        finally:
            self.ai_thinking = False
    
    def start_game(self, human_color):
        """Start a game with human player"""
        self.board = chess.Board()
        self.human_color = human_color
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.ai_thinking = False
        
        self.update_board_display()
        self.update_move_history()
        
        color_name = "White" if human_color == chess.WHITE else "Black"
        self.status_var.set(f"You are playing as {color_name}. Click a piece to move.")
        
        if human_color == chess.BLACK:
            self.window.after(500, self.ai_move)
    
    def watch_ai_game(self):
        """Watch AI play against itself"""
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.ai_thinking = False
        
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("AI vs AI demo started")
        
        self.play_ai_vs_ai()
    
    def play_ai_vs_ai(self):
        """Play AI vs AI game with visualization"""
        if not self.board.is_game_over():
            try:
                move = self.ai.select_move(self.board, temperature=0.1)
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
        """Reset the game"""
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.legal_moves_for_selected = []
        self.move_history = []
        self.ai_thinking = False
        
        self.update_board_display()
        self.update_move_history()
        self.status_var.set("Game reset - Choose a play mode")
    
    def game_over(self):
        """Handle game over"""
        result = self.board.result()
        outcome = self.board.outcome()
        
        if outcome is None:
            message = "Game ended"
        elif outcome.winner == chess.WHITE:
            message = "White wins!"
        elif outcome.winner == chess.BLACK:
            message = "Black wins!"
        else:
            message = "Draw!"
        
        # Add termination reason
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
        """Process messages from training thread"""
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
        """Callback during training - called from training thread"""
        result = "Win" if reward > 0 else ("Loss" if reward < 0 else "Draw")
        text = f"Game {game_num}/{total_games}\n{result} (Reward: {reward:.2f})\nPolicy Loss: {policy_loss:.4f}\nValue Loss: {value_loss:.4f}"
        
        self.message_queue.put({
            'type': 'training_update',
            'text': text
        })
    
    def train_worker(self, num_games, temperature):
        """Worker function for training thread"""
        try:
            self.ai.train(
                num_games=num_games, 
                temperature=temperature, 
                callback=self.training_callback
            )
            
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
        """Start training process in a separate thread"""
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
        
        # Start training in a separate thread
        self.training_thread = threading.Thread(
            target=self.train_worker,
            args=(num_games, temperature),
            daemon=True
        )
        self.training_thread.start()
    
    def stop_training(self):
        """Stop training process"""
        if self.is_training:
            self.ai.stop_training()
            self.status_var.set("Stopping training...")
            # Wait a bit for training to stop
            self.window.after(1000, self.finish_stop_training)
        else:
            self.finish_stop_training()
    
    def finish_stop_training(self):
        """Finish stopping the training process"""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ai.save_model()
        self.update_stats_display()
        self.progress_var.set("Training stopped")
        self.status_var.set("Ready")
    
    def run(self):
        """Run the GUI"""
        self.window.mainloop()


if __name__ == "__main__":
    print("Chess AI - Self-Play Reinforcement Learning (IMPROVED VERSION)")
    print("=" * 60)
    print("Required packages: python-chess, torch, pillow, numpy")
    print("Install with: pip install python-chess torch pillow numpy")
    print("=" * 60)
    print("\nIMPROVEMENTS:")
    print("✓ Threading - Training won't freeze the GUI")
    print("✓ Visual feedback - Selected squares and legal moves highlighted")
    print("✓ Move history - See all moves in SAN notation")
    print("✓ Better error handling - More robust and informative")
    print("✓ Improved board rendering - Better visuals with coordinates")
    print("✓ Stop training - Actually works now!")
    print("=" * 60)
    
    gui = ChessGUI()
    gui.run()