# Chess AI - Self-Play Reinforcement Learning

A chess AI that learns through self-play using deep reinforcement learning. The AI uses a neural network to evaluate positions and suggest moves, improving over time by playing games against itself.

## Features

- **Self-Play Training**: AI plays against itself to learn chess strategies
- **Neural Network**: Uses a convolutional neural network with policy and value heads
- **GUI Interface**: Simple GUI built with tkinter for easy interaction
- **Play Against AI**: Challenge the AI after it has trained
- **Watch AI vs AI**: Observe the AI play against itself
- **Save/Load Progress**: Automatically saves training progress and model weights
- **Training Statistics**: Track wins, losses, draws, and total games played

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Setup Steps

1. **Install required packages:**
```bash
pip install -r requirements.txt
```

Or install manually:
```bash
pip install python-chess torch pillow numpy
```

2. **Run the program:**
```bash
python chess_ai_selfplay.py
```

## Usage

### GUI Controls

The GUI has three main sections:

#### 1. Training Controls
- **Number of games**: Set how many self-play games to run
- **Start Training**: Begin the training process (AI plays against itself)
- **Stop Training**: Stop training and save progress

#### 2. Play Mode
- **Play as White**: Start a game where you play as white
- **Play as Black**: Start a game where you play as black
- **AI vs AI**: Watch the AI play against itself (no training)
- **New Game**: Reset the board for a new game

#### 3. Statistics Display
Shows training progress including:
- Games played
- Total moves
- White/Black wins
- Draws
- Model save location

### How to Train the AI

1. Launch the program
2. Set the number of games (start with 10-50 for testing)
3. Click "Start Training"
4. The AI will play games against itself and learn
5. Training progress is automatically saved every 10 games
6. You can stop training at any time and resume later

### How to Play Against the AI

1. Click "Play as White" or "Play as Black"
2. Click on a piece to select it
3. Click on a destination square to move
4. The AI will respond with its move
5. Continue until the game ends

## How It Works

### Neural Network Architecture

The AI uses a convolutional neural network with:
- **Input**: 8×8×12 board representation (6 piece types × 2 colors)
- **Convolutional layers**: Extract spatial features from the board
- **Policy head**: Outputs move probabilities (4096 possible from-to moves)
- **Value head**: Outputs position evaluation (-1 to +1)

### Training Process

1. **Self-Play**: The AI plays complete games against itself
2. **Data Collection**: Each move and board position is recorded
3. **Reward Assignment**: +1 for win, -1 for loss, 0 for draw
4. **Network Update**: The network learns to:
   - Choose moves that lead to wins (policy)
   - Evaluate positions accurately (value)

### Model Persistence

Models are saved in the `chess_ai_models` directory:
- `model_latest.pth`: Most recent model (automatically loaded on startup)
- `model_YYYYMMDD_HHMMSS.pth`: Timestamped backups

## Training Tips

1. **Start Small**: Begin with 10-50 games to verify everything works
2. **Gradual Training**: Train in batches (50-100 games at a time)
3. **Patience**: The AI needs hundreds of games to become competent
4. **Regular Saves**: Training auto-saves every 10 games
5. **Testing**: Periodically play against the AI to see improvement

## Expected Training Time

Training time depends on your hardware:
- **CPU only**: ~30-60 seconds per game
- **GPU**: ~5-15 seconds per game

For reasonable chess ability:
- **Beginner level**: 100-500 games
- **Intermediate level**: 1,000-5,000 games
- **Advanced level**: 10,000+ games

## Limitations

This is a simplified implementation for educational purposes:
- Uses a basic policy gradient approach (not full AlphaZero)
- No Monte Carlo Tree Search (MCTS)
- Limited board evaluation depth
- May not reach grandmaster level without significant training

## Troubleshooting

### Memory Issues
If training runs out of memory:
- Reduce the number of games per training session
- Close other applications
- Consider training on a machine with more RAM

### Slow Training
To speed up training:
- Use a GPU if available (PyTorch will automatically detect it)
- Reduce the number of games in the GUI to smaller batches
- Train overnight for longer sessions

## Future Improvements

Potential enhancements:
- Add Monte Carlo Tree Search (MCTS)
- Implement opening book
- Add endgame tablebase
- Improve neural network architecture
- Add distributed training
- Implement ELO rating system
- Add game analysis features

## Technical Details

### Files Created
- `chess_ai_models/`: Directory containing saved models and training data
  - `model_latest.pth`: Current model weights
  - `model_*.pth`: Backup models with timestamps

### Model Structure
```python
ChessNet(
  conv1: Conv2d(12, 128, kernel_size=3, padding=1)
  conv2: Conv2d(128, 128, kernel_size=3, padding=1)
  conv3: Conv2d(128, 128, kernel_size=3, padding=1)
  fc1: Linear(8192, 512)
  policy_fc: Linear(512, 4096)  # Move probabilities
  value_fc1: Linear(512, 128)   # Position evaluation
  value_fc2: Linear(128, 1)
)
```

## License

This is an educational project. Feel free to modify and experiment!

## Credits

Built using:
- [python-chess](https://python-chess.readthedocs.io/) for chess game logic
- [PyTorch](https://pytorch.org/) for neural network
- [tkinter](https://docs.python.org/3/library/tkinter.html) for GUI
- [Pillow](https://pillow.readthedocs.io/) for board rendering

Inspired by AlphaZero and other self-play reinforcement learning approaches.
