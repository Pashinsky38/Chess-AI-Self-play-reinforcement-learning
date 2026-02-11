"""
Chess AI with Self-Play Reinforcement Learning
Uses a neural network that learns by playing against itself
"""

import chess
import chess.svg
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
import pickle
import os
from datetime import datetime
import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageTk
import io
import cairosvg

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
        
        while not board.is_game_over():
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
        for game_num in range(num_games):
            # Play a game
            game_data, reward = self.play_game(temperature)
            
            # Train on the game
            policy_loss, value_loss = self.train_on_game(game_data, reward)
            
            if callback:
                callback(game_num + 1, num_games, policy_loss, value_loss, reward)
            
            # Save model periodically
            if (game_num + 1) % 10 == 0:
                self.save_model()
    
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
        
        # Also save a timestamped backup
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
            checkpoint = torch.load(model_path, map_location=self.device)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.training_stats = checkpoint['training_stats']
            print(f"Model loaded from {model_path}")
            print(f"Training stats: {self.training_stats}")
        else:
            print("No saved model found. Starting fresh.")


class ChessGUI:
    def __init__(self):
        self.window = tk.Tk()
        self.window.title("Chess AI - Self-Play Reinforcement Learning")
        self.window.geometry("900x700")
        
        self.ai = ChessAI()
        self.board = chess.Board()
        self.selected_square = None
        self.human_color = None
        self.is_training = False
        self.training_thread = None
        
        self.setup_gui()
        
    def setup_gui(self):
        """Setup the GUI components"""
        # Main container
        main_frame = ttk.Frame(self.window, padding="10")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Board display
        self.board_label = ttk.Label(main_frame)
        self.board_label.grid(row=0, column=0, columnspan=2, pady=10)
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="Controls", padding="10")
        control_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        # Training controls
        train_frame = ttk.Frame(control_frame)
        train_frame.grid(row=0, column=0, columnspan=2, pady=5)
        
        ttk.Label(train_frame, text="Number of games:").grid(row=0, column=0, padx=5)
        self.num_games_var = tk.StringVar(value="10")
        ttk.Entry(train_frame, textvariable=self.num_games_var, width=10).grid(row=0, column=1, padx=5)
        
        self.train_button = ttk.Button(train_frame, text="Start Training", command=self.start_training)
        self.train_button.grid(row=0, column=2, padx=5)
        
        self.stop_button = ttk.Button(train_frame, text="Stop Training", command=self.stop_training, state=tk.DISABLED)
        self.stop_button.grid(row=0, column=3, padx=5)
        
        # Play against AI controls
        play_frame = ttk.Frame(control_frame)
        play_frame.grid(row=1, column=0, columnspan=2, pady=5)
        
        ttk.Button(play_frame, text="Play as White", command=lambda: self.start_game(chess.WHITE)).grid(row=0, column=0, padx=5)
        ttk.Button(play_frame, text="Play as Black", command=lambda: self.start_game(chess.BLACK)).grid(row=0, column=1, padx=5)
        ttk.Button(play_frame, text="AI vs AI", command=self.watch_ai_game).grid(row=0, column=2, padx=5)
        ttk.Button(play_frame, text="New Game", command=self.reset_game).grid(row=0, column=3, padx=5)
        
        # Stats display
        stats_frame = ttk.LabelFrame(main_frame, text="Training Statistics", padding="10")
        stats_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=10)
        
        self.stats_text = tk.Text(stats_frame, height=8, width=70)
        self.stats_text.grid(row=0, column=0)
        
        # Status bar
        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(main_frame, textvariable=self.status_var, relief=tk.SUNKEN).grid(row=3, column=0, columnspan=2, sticky=(tk.W, tk.E))
        
        self.update_board_display()
        self.update_stats_display()
        
        # Bind click event for board
        self.board_label.bind("<Button-1>", self.on_board_click)
    
    def board_to_image(self):
        """Convert chess board to image"""
        svg_data = chess.svg.board(board=self.board, size=500)
        png_data = cairosvg.svg2png(bytestring=svg_data.encode('utf-8'))
        image = Image.open(io.BytesIO(png_data))
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
        stats_text = f"""
Games Played: {stats['games_played']}
Total Moves: {stats['total_moves']}
White Wins: {stats['white_wins']}
Black Wins: {stats['black_wins']}
Draws: {stats['draws']}

Model Directory: {self.ai.save_dir}
        """.strip()
        
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
    
    def on_board_click(self, event):
        """Handle board click for human moves"""
        if self.human_color is None or self.board.turn != self.human_color:
            return
        
        if self.board.is_game_over():
            return
        
        # Calculate which square was clicked (approximate)
        # This is a simplified version - proper implementation would need exact coordinate mapping
        x, y = event.x, event.y
        square_size = 500 // 8
        file = min(7, max(0, x // square_size))
        rank = min(7, max(0, 7 - (y // square_size)))
        square = chess.square(file, rank)
        
        if self.selected_square is None:
            # Select piece
            piece = self.board.piece_at(square)
            if piece and piece.color == self.human_color:
                self.selected_square = square
                self.status_var.set(f"Selected {chess.SQUARE_NAMES[square]}")
        else:
            # Try to make move
            move = chess.Move(self.selected_square, square)
            
            # Check for promotion
            if move in self.board.legal_moves or chess.Move(self.selected_square, square, promotion=chess.QUEEN) in self.board.legal_moves:
                if self.board.piece_at(self.selected_square).piece_type == chess.PAWN and (rank == 0 or rank == 7):
                    move = chess.Move(self.selected_square, square, promotion=chess.QUEEN)
                
                if move in self.board.legal_moves:
                    self.board.push(move)
                    self.update_board_display()
                    self.selected_square = None
                    
                    if self.board.is_game_over():
                        self.game_over()
                    else:
                        # AI's turn
                        self.window.after(500, self.ai_move)
                else:
                    self.status_var.set("Illegal move!")
                    self.selected_square = None
            else:
                self.status_var.set("Illegal move!")
                self.selected_square = None
    
    def ai_move(self):
        """Make AI move"""
        if self.board.is_game_over():
            self.game_over()
            return
        
        self.status_var.set("AI thinking...")
        self.window.update()
        
        move = self.ai.select_move(self.board, temperature=0.1)
        self.board.push(move)
        
        self.status_var.set(f"AI played: {move}")
        self.update_board_display()
        
        if self.board.is_game_over():
            self.game_over()
    
    def start_game(self, human_color):
        """Start a game with human player"""
        self.board = chess.Board()
        self.human_color = human_color
        self.selected_square = None
        self.update_board_display()
        
        color_name = "White" if human_color == chess.WHITE else "Black"
        self.status_var.set(f"You are playing as {color_name}")
        
        if human_color == chess.BLACK:
            self.window.after(500, self.ai_move)
    
    def watch_ai_game(self):
        """Watch AI play against itself"""
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.update_board_display()
        self.status_var.set("AI vs AI game started")
        
        self.play_ai_vs_ai()
    
    def play_ai_vs_ai(self):
        """Play AI vs AI game with visualization"""
        if not self.board.is_game_over():
            move = self.ai.select_move(self.board, temperature=0.1)
            self.board.push(move)
            self.update_board_display()
            self.status_var.set(f"Move: {move}")
            self.window.after(500, self.play_ai_vs_ai)
        else:
            self.game_over()
    
    def reset_game(self):
        """Reset the game"""
        self.board = chess.Board()
        self.human_color = None
        self.selected_square = None
        self.update_board_display()
        self.status_var.set("New game started")
    
    def game_over(self):
        """Handle game over"""
        result = self.board.result()
        outcome = self.board.outcome()
        
        if outcome.winner == chess.WHITE:
            message = "White wins!"
        elif outcome.winner == chess.BLACK:
            message = "Black wins!"
        else:
            message = "Draw!"
        
        self.status_var.set(f"Game Over: {message} ({result})")
        messagebox.showinfo("Game Over", message)
    
    def training_callback(self, game_num, total_games, policy_loss, value_loss, reward):
        """Callback during training"""
        self.status_var.set(f"Training: Game {game_num}/{total_games} - Reward: {reward:.2f}")
        self.update_stats_display()
        self.window.update()
    
    def start_training(self):
        """Start training process"""
        try:
            num_games = int(self.num_games_var.get())
        except ValueError:
            messagebox.showerror("Error", "Please enter a valid number of games")
            return
        
        self.is_training = True
        self.train_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.status_var.set("Training started...")
        
        # Train in the main thread (simplified version)
        # For production, use threading to avoid blocking GUI
        try:
            self.ai.train(num_games=num_games, temperature=1.0, callback=self.training_callback)
            self.status_var.set("Training completed!")
            messagebox.showinfo("Training Complete", f"Trained on {num_games} games")
        except Exception as e:
            messagebox.showerror("Training Error", str(e))
        finally:
            self.stop_training()
    
    def stop_training(self):
        """Stop training process"""
        self.is_training = False
        self.train_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.ai.save_model()
        self.update_stats_display()
    
    def run(self):
        """Run the GUI"""
        self.window.mainloop()


if __name__ == "__main__":
    print("Chess AI - Self-Play Reinforcement Learning")
    print("=" * 50)
    print("Required packages: python-chess, torch, pillow, cairosvg")
    print("Install with: pip install python-chess torch pillow cairosvg")
    print("=" * 50)
    
    gui = ChessGUI()
    gui.run()
