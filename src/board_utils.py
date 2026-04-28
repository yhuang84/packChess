import chess
import numpy as np
from typing import Tuple, Optional

BOARD_PLANES = 14
POLICY_PLANES = 73
BOARD_SIZE = 8
NUM_POLICY_CLASSES = POLICY_PLANES * BOARD_SIZE * BOARD_SIZE  # 4672

# Pre-computed move offsets for faster lookups
PROMOTION_TO_OFFSET = {
    chess.KNIGHT: 0,
    chess.BISHOP: 1,
    chess.ROOK: 2,
}

KNIGHT_MOVES = [
    (2, 1), (1, 2), (-1, 2), (-2, 1),
    (-2, -1), (-1, -2), (1, -2), (2, -1),
]

QUEEN_DIRECTIONS = [
    (1, 0),   # N
    (1, 1),   # NE
    (0, 1),   # E
    (-1, 1),  # SE
    (-1, 0),  # S
    (-1, -1), # SW
    (0, -1),  # W
    (1, -1),  # NW
]


def move_to_index(move: chess.Move) -> Tuple[int, int, int]:
    """
    Map a chess.Move to (plane, row, col) in a 73x8x8 policy tensor.

    This keeps the standard action encoding style:
    - planes 0-55: queen-like directions x distances
    - planes 56-63: knight moves
    - planes 64-72: underpromotions

    Returns:
        tuple[int, int, int]: (plane, from_row, from_col)
        plane == -1 means the move could not be encoded.
    """
    from_square = move.from_square
    to_square = move.to_square

    from_row = chess.square_rank(from_square)
    from_col = chess.square_file(from_square)
    to_row = chess.square_rank(to_square)
    to_col = chess.square_file(to_square)

    dr = to_row - from_row
    dc = to_col - from_col
    plane = -1

    # Underpromotions: knight / bishop / rook across left / straight / right
    if move.promotion and move.promotion != chess.QUEEN:
        if move.promotion in PROMOTION_TO_OFFSET and dc in (-1, 0, 1):
            plane = 64 + (PROMOTION_TO_OFFSET[move.promotion] * 3) + (dc + 1)

    # Knight moves
    elif abs(dr * dc) == 2:
        if (dr, dc) in KNIGHT_MOVES:
            plane = 56 + KNIGHT_MOVES.index((dr, dc))

    # Queen-like moves
    else:
        for d_idx, (r_dir, c_dir) in enumerate(QUEEN_DIRECTIONS):
            for dist in range(1, 8):
                if dr == r_dir * dist and dc == c_dir * dist:
                    plane = (d_idx * 7) + (dist - 1)
                    break
            if plane != -1:
                break

    return plane, from_row, from_col


def move_to_class_index(move: chess.Move) -> int:
    """
    Flatten the 73x8x8 move encoding into a single class index in [0, 4671].

    This is much cheaper to train with CrossEntropyLoss than building a full
    one-hot 73x8x8 target tensor for every sample.
    """
    plane, row, col = move_to_index(move)
    if plane < 0:
        return -1
    return plane * 64 + row * 8 + col


def result_to_value(result: str) -> Optional[float]:
    """
    Convert PGN Result header into a scalar value from White's perspective.

    Returns:
        1.0 for White win
        0.0 for draw
       -1.0 for Black win
        None for unknown / unfinished games
    """
    mapping = {
        "1-0": 1.0,
        "1/2-1/2": 0.0,
        "0-1": -1.0,
    }
    return mapping.get(result)


def board_to_tensor(board: chess.Board) -> np.ndarray:
    """
    Convert a python-chess board into a (14, 8, 8) float32 tensor.

    Planes 0-5: White pieces  (P, N, B, R, Q, K)
    Planes 6-11: Black pieces (P, N, B, R, Q, K)
    Plane 12: Side to move (all 1s if White to move)
    Plane 13: Castling-rights markers in the four corners
    """
    tensor = np.zeros((BOARD_PLANES, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)

    piece_types = [
        chess.PAWN,
        chess.KNIGHT,
        chess.BISHOP,
        chess.ROOK,
        chess.QUEEN,
        chess.KING,
    ]

    for i, piece_type in enumerate(piece_types):
        for square in board.pieces(piece_type, chess.WHITE):
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[i, row, col] = 1.0

        for square in board.pieces(piece_type, chess.BLACK):
            row = chess.square_rank(square)
            col = chess.square_file(square)
            tensor[i + 6, row, col] = 1.0

    if board.turn == chess.WHITE:
        tensor[12, :, :] = 1.0

    if board.has_kingside_castling_rights(chess.WHITE):
        tensor[13, 0, 7] = 1.0
    if board.has_queenside_castling_rights(chess.WHITE):
        tensor[13, 0, 0] = 1.0
    if board.has_kingside_castling_rights(chess.BLACK):
        tensor[13, 7, 7] = 1.0
    if board.has_queenside_castling_rights(chess.BLACK):
        tensor[13, 7, 0] = 1.0

    return tensor


if __name__ == "__main__":
    test_board = chess.Board()
    x = board_to_tensor(test_board)
    print("Board tensor shape:", x.shape)
    print("Policy classes:", NUM_POLICY_CLASSES)
    print("e2e4 class index:", move_to_class_index(chess.Move.from_uci("e2e4")))