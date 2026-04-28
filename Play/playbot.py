import os
import sys
import pygame
import chess
import torch

# -----------------------------------------------------------------------------
# Project imports & Paths
# -----------------------------------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
SRC_DIR = os.path.abspath(os.path.join(ROOT_DIR, 'src'))

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

from network import ChessNet
from mcts import MCTS

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
SQ_SIZE = 150  # Increased size for a larger 768x768 window
BOARD_SIZE = SQ_SIZE * 8
COLORS = [pygame.Color(240, 217, 181), pygame.Color(181, 136, 99)]
SELECT_COLOR = (246, 246, 105, 130)

IMAGES = {}

def load_images() -> None:
    pieces = ['wP', 'wR', 'wN', 'wB', 'wQ', 'wK', 'bP', 'bR', 'bN', 'bB', 'bQ', 'bK']
    png_dir = os.path.join(ROOT_DIR, "PNG")
    for piece in pieces:
        img_path = os.path.join(png_dir, f"{piece}.png")
        IMAGES[piece] = pygame.transform.smoothscale(pygame.image.load(img_path), (SQ_SIZE, SQ_SIZE))

# -----------------------------------------------------------------------------
# Main GUI Loop
# -----------------------------------------------------------------------------
def main(model_path: str) -> None:
    pygame.init()
    screen = pygame.display.set_mode((BOARD_SIZE, BOARD_SIZE))
    pygame.display.set_caption("PackChess Bot")
    clock = pygame.time.Clock()
    
    load_images()

    # Hardware detection
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading engine on {device}...")
    
    # Initialize Model
    model = ChessNet().to(device)
    checkpoint = torch.load(model_path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict)
    model.eval()

    # Initialize MCTS (400 simulations per move for real-time play)
    mcts = MCTS(model, device, num_simulations=400)

    board = chess.Board()
    selected_square = None
    running = True
    game_over = False

    while running:
        # Check game over state to update the title
        if not game_over and board.is_game_over():
            outcome = board.outcome()
            result_text = "Draw"
            if outcome is not None and outcome.winner == chess.WHITE:
                result_text = "White Wins!"
            elif outcome is not None and outcome.winner == chess.BLACK:
                result_text = "Black Wins!"
            
            pygame.display.set_caption(f"PackChess Bot - Game Over: {result_text}")
            game_over = True

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            
            # Human turn (White)
            elif event.type == pygame.MOUSEBUTTONDOWN and board.turn == chess.WHITE and not game_over:
                col = pygame.mouse.get_pos()[0] // SQ_SIZE
                row = pygame.mouse.get_pos()[1] // SQ_SIZE
                clicked_square = chess.square(col, 7 - row)

                if selected_square is None:
                    piece = board.piece_at(clicked_square)
                    if piece and piece.color == chess.WHITE:
                        selected_square = clicked_square
                else:
                    # Construct move (auto-queen for simplicity)
                    move = chess.Move(selected_square, clicked_square)
                    piece = board.piece_at(selected_square)
                    if piece and piece.piece_type == chess.PAWN and chess.square_rank(clicked_square) == 7:
                        move = chess.Move(selected_square, clicked_square, promotion=chess.QUEEN)

                    if move in board.legal_moves:
                        board.push(move)
                    
                    selected_square = None # Deselect after move attempt

        # Bot turn (Black)
        if board.turn == chess.BLACK and not game_over:
            pygame.display.set_caption("PackChess Bot - Thinking...")
            pygame.event.pump() # Keep UI responsive
            
            # Call the MCTS search instead of pure policy
            best_move = mcts.search(board)
            board.push(best_move)
            
            if not board.is_game_over():
                pygame.display.set_caption("PackChess Bot")

        # Rendering
        for r in range(8):
            for c in range(8):
                pygame.draw.rect(screen, COLORS[(r + c) % 2], (c * SQ_SIZE, r * SQ_SIZE, SQ_SIZE, SQ_SIZE))

        if selected_square is not None:
            col = chess.square_file(selected_square)
            row = 7 - chess.square_rank(selected_square)
            s = pygame.Surface((SQ_SIZE, SQ_SIZE), pygame.SRCALPHA)
            s.fill(SELECT_COLOR)
            screen.blit(s, (col * SQ_SIZE, row * SQ_SIZE))

        for square in chess.SQUARES:
            piece = board.piece_at(square)
            if piece:
                img_key = ('w' if piece.color == chess.WHITE else 'b') + piece.symbol().upper()
                col = chess.square_file(square)
                row = 7 - chess.square_rank(square)
                screen.blit(IMAGES[img_key], (col * SQ_SIZE, row * SQ_SIZE))

        pygame.display.flip()
        clock.tick(30)

    pygame.quit()

if __name__ == "__main__":
    # Ensure it looks in the correct V5 Large checkpoints folder
    CHECKPOINT_DIR = os.path.abspath(os.path.join(ROOT_DIR, "checkpoints_V5_Large"))
    default_model = os.path.join(CHECKPOINT_DIR, "chess_model_epoch_15_V5.pth")
    
    # Fallback to .pt if .pth is not found
    if not os.path.exists(default_model):
        default_model = os.path.join(CHECKPOINT_DIR, "chess_model_epoch_15_V5.pt")

    model_weights = sys.argv[1] if len(sys.argv) > 1 else default_model
    
    if not os.path.exists(model_weights):
        print(f"Error: Model not found at {model_weights}")
    else:
        main(model_weights)