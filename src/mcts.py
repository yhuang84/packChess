import math
from typing import Dict, List, Tuple, Optional

import chess
import numpy as np
import torch

from board_utils import board_to_tensor, move_to_class_index


# -----------------------------------------------------------------------------
# Classical eval settings (Only active if use_heuristics=True)
# -----------------------------------------------------------------------------

PIECE_VALUES = {
    chess.PAWN: 100,
    chess.KNIGHT: 320,
    chess.BISHOP: 330,
    chess.ROOK: 500,
    chess.QUEEN: 900,
    chess.KING: 0,
}

# Heuristic blending weights
NN_VALUE_WEIGHT_HEURISTIC = 0.60
MATERIAL_VALUE_WEIGHT_HEURISTIC = 0.40

# Converts material centipawns to [-1, 1].
MATERIAL_SCALE_CP = 1200.0

# Forcing move boosts.
PAWN_CAPTURE_BOOST = 2.0
MINOR_CAPTURE_BOOST = 8.0
ROOK_CAPTURE_BOOST = 40.0
QUEEN_CAPTURE_BOOST = 60.0

CHECK_PRIOR_BOOST = 1.25
PROMOTION_PRIOR_BOOST = 12.0


def clamp(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def material_balance_for(board: chess.Board, color: chess.Color) -> int:
    score = 0
    for piece in board.piece_map().values():
        value = PIECE_VALUES[piece.piece_type]
        score += value if piece.color == color else -value
    return score


def material_eval_side_to_move(board: chess.Board) -> float:
    cp = material_balance_for(board, board.turn)
    return clamp(cp / MATERIAL_SCALE_CP)


def terminal_value_side_to_move(board: chess.Board) -> Optional[float]:
    if not (board.is_game_over(claim_draw=True) or board.can_claim_draw()):
        return None

    outcome = board.outcome(claim_draw=True)

    if outcome is None or outcome.winner is None:
        return 0.0

    return 1.0 if outcome.winner == board.turn else -1.0


def has_mate_in_one(board: chess.Board) -> Optional[chess.Move]:
    for move in board.legal_moves:
        board.push(move)
        is_mate = board.is_checkmate()
        board.pop()
        if is_mate:
            return move
    return None


def capture_prior_multiplier(board: chess.Board, move: chess.Move) -> float:
    if not board.is_capture(move):
        return 1.0

    captured_piece = board.piece_at(move.to_square)

    if captured_piece is None and board.is_en_passant(move):
        return PAWN_CAPTURE_BOOST

    if captured_piece is None:
        return 1.0

    if captured_piece.piece_type == chess.QUEEN:
        return QUEEN_CAPTURE_BOOST
    if captured_piece.piece_type == chess.ROOK:
        return ROOK_CAPTURE_BOOST
    if captured_piece.piece_type in (chess.BISHOP, chess.KNIGHT):
        return MINOR_CAPTURE_BOOST
    if captured_piece.piece_type == chess.PAWN:
        return PAWN_CAPTURE_BOOST

    return 1.0


def move_gives_check(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    gives_check = board.is_check()
    board.pop()
    return gives_check


class Node:
    def __init__(self, prior_prob: float):
        self.visit_count = 0
        self.value_sum = 0.0
        self.prior_prob = prior_prob
        self.children: Dict[chess.Move, "Node"] = {}
        self.virtual_loss = 0

    @property
    def q_value(self) -> float:
        effective_visits = self.visit_count + self.virtual_loss
        if effective_visits == 0:
            return 0.0
        return (self.value_sum - self.virtual_loss) / effective_visits

    def expand(self, action_probs: Dict[chess.Move, float]):
        for move, prob in action_probs.items():
            if move not in self.children:
                self.children[move] = Node(prior_prob=prob)

    def is_expanded(self) -> bool:
        return len(self.children) > 0


class MCTS:
    def __init__(
        self,
        model: torch.nn.Module,
        device: torch.device,
        num_simulations: int = 400,
        c_puct: float = 1.5,
        batch_size: int = 16,
        use_heuristics: bool = True, 
    ):
        self.model = model
        self.device = device
        self.num_simulations = num_simulations
        self.c_puct = c_puct
        self.batch_size = batch_size
        self.use_heuristics = use_heuristics

        self.last_root_stats: List[Tuple[chess.Move, int, float, float]] = []

    @torch.no_grad()
    def search(self, board: chess.Board) -> chess.Move:
        ranked = self.search_ranked(board)
        if not ranked:
            return list(board.legal_moves)[0]
        return ranked[0][0]

    @torch.no_grad()
    def search_ranked(self, board: chess.Board) -> List[Tuple[chess.Move, int, float, float]]:
        root = Node(prior_prob=1.0)
        simulations_completed = 0

        root_mate = has_mate_in_one(board)
        if root_mate is not None:
            self.last_root_stats = [(root_mate, self.num_simulations, 1.0, 1.0)]
            return self.last_root_stats

        while simulations_completed < self.num_simulations:
            current_batch_size = min(
                self.batch_size,
                self.num_simulations - simulations_completed,
            )

            paths_to_evaluate = []
            boards_to_evaluate = []
            terminal_states = []

            # 1. Selection
            for _ in range(current_batch_size):
                node = root
                sim_board = board.copy(stack=False)
                search_path = [node]

                while node.is_expanded():
                    best_score = float("-inf")
                    best_action: Optional[chess.Move] = None
                    parent_visits = node.visit_count + node.virtual_loss

                    for action, child in node.children.items():
                        child_visits = child.visit_count + child.virtual_loss
                        q = -child.q_value if child_visits > 0 else (node.q_value - 0.1)
                        u = (
                            self.c_puct
                            * child.prior_prob
                            * math.sqrt(parent_visits + 1e-8)
                            / (1 + child_visits)
                        )

                        score = q + u

                        if score > best_score:
                            best_score = score
                            best_action = action

                    if best_action is None:
                        break

                    node = node.children[best_action]
                    sim_board.push(best_action)
                    search_path.append(node)
                    node.virtual_loss += 1

                terminal_value = terminal_value_side_to_move(sim_board)

                if terminal_value is not None:
                    terminal_states.append((search_path, terminal_value))
                else:
                    paths_to_evaluate.append(search_path)
                    boards_to_evaluate.append(sim_board)

            # 2. Batched NN evaluation and expansion
            if boards_to_evaluate:
                tensors = [board_to_tensor(b) for b in boards_to_evaluate]
                batch_tensor = torch.as_tensor(
                    np.array(tensors),
                    dtype=torch.float32,
                    device=self.device,
                )

                policy_out, value_out = self.model(batch_tensor)
                policy_np = policy_out.detach().cpu().numpy()
                value_np = value_out.detach().cpu().numpy()

                for i, sim_board in enumerate(boards_to_evaluate):
                    search_path = paths_to_evaluate[i]
                    node = search_path[-1]

                    single_policy = policy_np[i]
                    if single_policy.ndim == 2:
                        single_policy = single_policy.flatten()

                    nn_value = float(value_np[i][0]) if value_np.ndim == 2 else float(value_np[i])
                    nn_value = clamp(nn_value)

                    # Toggle between Pure NN and Heuristic blend
                    if self.use_heuristics:
                        mat_value = material_eval_side_to_move(sim_board)
                        value = (
                            NN_VALUE_WEIGHT_HEURISTIC * nn_value
                            + MATERIAL_VALUE_WEIGHT_HEURISTIC * mat_value
                        )
                    else:
                        value = nn_value

                    value = clamp(value)
                    legal_moves = list(sim_board.legal_moves)

                    if not legal_moves:
                        self._backpropagate(search_path, value)
                        continue

                    mate_move = has_mate_in_one(sim_board)
                    if mate_move is not None:
                        node.expand({mate_move: 1.0})
                        self._backpropagate(search_path, 1.0)
                        continue

                    valid_moves = [
                        move for move in legal_moves
                        if move_to_class_index(move) != -1
                    ]

                    action_probs: Dict[chess.Move, float] = {}
                    prob_sum = 0.0

                    if valid_moves:
                        max_logit = max(
                            float(single_policy[move_to_class_index(move)])
                            for move in valid_moves
                        )

                        for move in valid_moves:
                            idx = move_to_class_index(move)
                            prob = math.exp(float(single_policy[idx]) - max_logit)

                            # Apply cheap tactical priors ONLY if heuristics are enabled
                            if self.use_heuristics:
                                prob *= capture_prior_multiplier(sim_board, move)
                                if move.promotion is not None:
                                    prob *= PROMOTION_PRIOR_BOOST
                                if move_gives_check(sim_board, move):
                                    prob *= CHECK_PRIOR_BOOST

                            action_probs[move] = prob
                            prob_sum += prob

                    if prob_sum <= 0.0:
                        uniform = 1.0 / len(legal_moves)
                        action_probs = {move: uniform for move in legal_moves}
                    else:
                        for move in action_probs:
                            action_probs[move] /= prob_sum

                    node.expand(action_probs)
                    self._backpropagate(search_path, value)

            # 3. Backprop terminal states
            for search_path, value in terminal_states:
                self._backpropagate(search_path, value)

            simulations_completed += current_batch_size

        if not root.children:
            fallback = list(board.legal_moves)[0]
            self.last_root_stats = [(fallback, 0, 0.0, 0.0)]
            return self.last_root_stats

        ranked = sorted(
            (
                (move, child.visit_count, child.q_value, child.prior_prob)
                for move, child in root.children.items()
            ),
            key=lambda item: item[1],
            reverse=True,
        )

        self.last_root_stats = ranked
        return ranked

    def _backpropagate(self, search_path: List[Node], value: float):
        value = clamp(value)
        
        DISCOUNT = 0.999  # <-- You need this to force the bot to seek mate!
        
        for node in reversed(search_path):
            if node.virtual_loss > 0:
                node.virtual_loss -= 1
            node.value_sum += value
            node.visit_count += 1
            
            # Flip the perspective AND apply the discount
            value = -value * DISCOUNT