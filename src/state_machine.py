import structlog
from typing import Dict, List, Optional
from schemas import Clue, ScoreAdjustment

logger = structlog.get_logger()


class TrebekStateMachine:
    """
    Advanced State Machine tracking quantitative board control to isolate
    Forrest Bouncing, calculates running game theory math securely, and defers
    "True Daily Double" calculations to runtime. Applies score adjustments sequentially.
    """

    def __init__(self, initial_scores: Optional[dict[str, int]] = None):
        self.scores: Dict[str, int] = initial_scores or {}
        self.pending_adjustments: List[ScoreAdjustment] = []
        self.current_board_control_contestant: Optional[str] = None

    def load_adjustments(self, adjustments: List[ScoreAdjustment]) -> None:
        """Loads score adjustments into the state machine."""
        self.pending_adjustments.extend(adjustments)

    def process_clue(self, clue: Clue) -> Clue:
        """
        Processes a single clue, updating scores based on responses, wagers,
        and chronologically anchored corrections. Tracks board control.
        """
        # Determine board control
        # We process responses chronologically
        if clue.round == "Jeopardy":
            clue_value = clue.board_row * 200
        elif clue.round == "Double Jeopardy":
            clue_value = clue.board_row * 400
        else:
            clue_value = 0

        # Handle True Daily Double Wager Interception
        if clue.is_daily_double and clue.daily_double_wager is not None and clue.wagerer_name:
            wagerer = clue.wagerer_name
            self.scores.setdefault(wagerer, 0)
            if clue.daily_double_wager == "True Daily Double":
                current_score = self.scores[wagerer]
                max_board_value = 1000 if clue.round == "Jeopardy" else 2000
                wager_amount = max(current_score, max_board_value)
                logger.info(
                    f"Resolved 'True Daily Double' for {wagerer} at index {clue.selection_order}: ${wager_amount}"
                )
                clue.daily_double_wager = wager_amount  # Store resolved value
            else:
                wager_amount = int(clue.daily_double_wager)
        else:
            wager_amount = clue_value

        for attempt in clue.attempts:
            player = attempt.speaker
            self.scores.setdefault(player, 0)

            if attempt.is_correct:
                self.scores[player] += wager_amount
                # Shift board control ONLY on correct response
                if self.current_board_control_contestant != player:
                    logger.info("Board control shift:", old=self.current_board_control_contestant, new=player)
                self.current_board_control_contestant = player
                break  # Only one person can be correct per clue
            else:
                self.scores[player] -= wager_amount

        # 3. Enforce Chronological Rigidity for Score Adjustments
        self._apply_score_adjustments_for_index(clue.selection_order)

        return clue

    def _apply_score_adjustments_for_index(self, current_index: int) -> None:
        """
        Checks for and applies any score corrections strictly anchored to the
        current clue selection index.
        """
        remaining_adjustments = []
        for adj in self.pending_adjustments:
            if adj.effective_after_clue_selection_order == current_index:
                self.scores.setdefault(adj.contestant, 0)
                self.scores[adj.contestant] += adj.points_adjusted
                logger.warning(
                    f"Applied anchored adjustment at index {current_index} for "
                    f"{adj.contestant}: {adj.points_adjusted > 0 and '+' or ''}{adj.points_adjusted} ({adj.reason})"
                )
            else:
                remaining_adjustments.append(adj)

        self.pending_adjustments = remaining_adjustments
