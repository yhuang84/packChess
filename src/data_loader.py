from __future__ import annotations

import math
import re
import random
import numpy as np
import chess
import chess.pgn

# Import your custom utility functions
from board_utils import BOARD_PLANES, board_to_tensor, move_to_class_index, result_to_value

# -----------------------------------------------------------------------------
# Value target settings
# -----------------------------------------------------------------------------
USE_STOCKFISH_EVALS = True
REQUIRE_STOCKFISH_EVAL = True
EVAL_CP_SCALE = 400.0
MATE_VALUE = 1.0

EVAL_RE = re.compile(r"\[%eval\s+([^\]\s]+)")

def parse_lichess_eval(comment: str):
    """
    Parse Lichess Stockfish eval comments.
    """
    if not comment:
        return None

    match = EVAL_RE.search(comment)
    if not match:
        return None

    token = match.group(1).strip()

    if token.startswith("#"):
        mate_text = token[1:]
        return -MATE_VALUE if mate_text.startswith("-") else MATE_VALUE

    try:
        pawns = float(token)
    except ValueError:
        return None

    centipawns = pawns * 100.0
    value = math.tanh(centipawns / EVAL_CP_SCALE)
    return float(max(-1.0, min(1.0, value)))

# -----------------------------------------------------------------------------
# Streamers & Generators
# -----------------------------------------------------------------------------

class PGNPositionStreamer:
    """
    Streams individual positions from a PGN file to allow mixing into a single batch.
    """
    def __init__(self, pgn_path: str, skip_unfinished: bool = True):
        self.pgn_path = pgn_path
        self.skip_unfinished = skip_unfinished

    def __iter__(self):
        with open(self.pgn_path, "r", encoding="utf-8", errors="replace") as pgn:
            while True:
                game = chess.pgn.read_game(pgn)
                if game is None:
                    break

                res_val = result_to_value(game.headers.get("Result", "*"))
                if self.skip_unfinished and res_val is None:
                    continue

                board = game.board()
                node = game
                
                while node.variations:
                    next_node = node.variation(0)
                    move = next_node.move
                    p_idx = move_to_class_index(move)

                    if p_idx < 0:
                        board.push(move)
                        node = next_node
                        continue

                    v_val = None
                    if USE_STOCKFISH_EVALS:
                        v_val = parse_lichess_eval(node.comment)
                    
                    if v_val is None and not REQUIRE_STOCKFISH_EVAL:
                        v_val = res_val

                    if v_val is not None:
                        current_val = v_val if board.turn == chess.WHITE else -v_val
                        yield board_to_tensor(board), p_idx, current_val

                    board.push(move)
                    node = next_node


class BlendedBatchGenerator:
    """
    Streams from two PGN files and mixes them into batches.
    """
    def __init__(self, game_path: str, tactics_path: str, batch_size: int = 1024, tactics_ratio: float = 0.2):
        self.game_path = game_path
        self.tactics_path = tactics_path
        self.batch_size = batch_size
        self.tactics_ratio = tactics_ratio

    def __iter__(self):
        game_stream = iter(PGNPositionStreamer(self.game_path))
        tactics_stream = iter(PGNPositionStreamer(self.tactics_path, skip_unfinished=False))
        
        num_tactics = int(self.batch_size * self.tactics_ratio)
        num_games = self.batch_size - num_tactics

        while True:
            x_batch = np.zeros((self.batch_size, BOARD_PLANES, 8, 8), dtype=np.float32)
            p_batch = np.zeros((self.batch_size,), dtype=np.int64)
            v_batch = np.zeros((self.batch_size, 1), dtype=np.float32)
            
            filled = 0
            try:
                for _ in range(num_tactics):
                    x, p, v = next(tactics_stream)
                    x_batch[filled], p_batch[filled], v_batch[filled, 0] = x, p, v
                    filled += 1
                
                for _ in range(num_games):
                    x, p, v = next(game_stream)
                    x_batch[filled], p_batch[filled], v_batch[filled, 0] = x, p, v
                    filled += 1
                
                indices = np.arange(self.batch_size)
                np.random.shuffle(indices)
                yield x_batch[indices], p_batch[indices], v_batch[indices]

            except StopIteration:
                break


class SingleStreamGenerator:
    """
    Replaces the old PGNBatchGenerator, using the new Streamer logic for non-blended runs.
    """
    def __init__(self, pgn_path: str, batch_size: int = 1024, skip_unfinished: bool = True):
        self.pgn_path = pgn_path
        self.batch_size = batch_size
        self.skip_unfinished = skip_unfinished

    def __iter__(self):
        stream = iter(PGNPositionStreamer(self.pgn_path, self.skip_unfinished))
        
        while True:
            x_batch = np.zeros((self.batch_size, BOARD_PLANES, 8, 8), dtype=np.float32)
            p_batch = np.zeros((self.batch_size,), dtype=np.int64)
            v_batch = np.zeros((self.batch_size, 1), dtype=np.float32)
            
            filled = 0
            try:
                for _ in range(self.batch_size):
                    x, p, v = next(stream)
                    x_batch[filled], p_batch[filled], v_batch[filled, 0] = x, p, v
                    filled += 1
                yield x_batch, p_batch, v_batch
            except StopIteration:
                if filled > 0:
                    yield x_batch[:filled], p_batch[:filled], v_batch[:filled]
                break


def generate_chess_batches(pgn_file_path: str, batch_size: int = 256, skip_unfinished: bool = True, tactics_path: str = None):
    """
    Public API used by trainer.py. Routes to the blended generator if tactics_path is provided.
    """
    if tactics_path:
        return BlendedBatchGenerator(
            game_path=pgn_file_path, 
            tactics_path=tactics_path, 
            batch_size=batch_size
        )
    else:
        return SingleStreamGenerator(
            pgn_path=pgn_file_path,
            batch_size=batch_size,
            skip_unfinished=skip_unfinished,
        )