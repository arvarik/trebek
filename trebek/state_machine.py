"""
Deterministic game state machine for J! episode verification.

Processes extracted clues sequentially, tracking running scores,
board control, Daily Double wager resolution, and chronologically
anchored score adjustments. Used as the final verification step
before committing relational data to the database.
"""

import structlog
from typing import Dict, List, Optional
from trebek.schemas import Clue, ScoreAdjustment, FinalJep

logger = structlog.get_logger()


class TrebekStateMachine:
    """
    Advanced State Machine tracking quantitative board control to isolate
    Forrest Bouncing, calculates running game theory math securely, and defers
    "True Daily Double" calculations to runtime. Applies score adjustments sequentially.
    """

    def __init__(
        self,
        initial_scores: Optional[Dict[str, int]] = None,
        valid_contestants: Optional[set[str]] = None,
    ):
        self.scores: Dict[str, int] = initial_scores or {}
        self.coryat_scores: Dict[str, int] = {}  # Clue face value only, no DD wagers, no FJ
        self.pending_adjustments: List[ScoreAdjustment] = []
        self.current_board_control_contestant: Optional[str] = None
        self.valid_contestants: Optional[set[str]] = valid_contestants
        self.unknown_speaker_warnings: int = 0

    def load_adjustments(self, adjustments: List[ScoreAdjustment]) -> None:
        """Loads score adjustments into the state machine."""
        self.pending_adjustments.extend(adjustments)

    def process_clue(self, clue: Clue) -> Clue:
        """
        Processes a single clue, updating scores based on responses, wagers,
        and chronologically anchored corrections. Tracks board control.
        """
        # 1. Determine clue value based on round
        if clue.round == "J!":
            clue_value = clue.board_row * 200
        elif clue.round == "Double J!":
            clue_value = clue.board_row * 400
        else:
            clue_value = 0

        # 2. Handle Daily Double — only one attempt allowed per J! rules
        if clue.is_daily_double and (clue.daily_double_wager is None or not clue.wagerer_name):
            logger.warning(
                "Daily Double missing wager or wagerer — falling back to standard scoring",
                category=clue.category,
                selection_order=clue.selection_order,
                has_wager=clue.daily_double_wager is not None,
                has_wagerer=bool(clue.wagerer_name),
            )
        if clue.is_daily_double and clue.daily_double_wager is not None and clue.wagerer_name:
            wagerer = clue.wagerer_name
            self.scores.setdefault(wagerer, 0)

            if clue.daily_double_wager == "True Daily Double":
                current_score = self.scores[wagerer]
                max_board_value = 1000 if clue.round == "J!" else 2000
                wager_amount = max(current_score, max_board_value)
                logger.info(
                    "Resolved 'True Daily Double'",
                    wagerer=wagerer,
                    selection_order=clue.selection_order,
                    wager_amount=wager_amount,
                )
                clue.daily_double_wager = wager_amount  # Store resolved value
            else:
                wager_amount = int(clue.daily_double_wager)

            # Daily Doubles: only the wagerer responds (max 1 attempt)
            if clue.attempts:
                attempt = clue.attempts[0]
                self.coryat_scores.setdefault(wagerer, 0)
                if attempt.is_correct:
                    self.scores[wagerer] += wager_amount
                    self.coryat_scores[wagerer] += clue_value  # Coryat uses face value
                    self.current_board_control_contestant = wagerer
                else:
                    self.scores[wagerer] -= wager_amount
                    self.coryat_scores[wagerer] -= clue_value  # Coryat uses face value
                    # Board control stays with wagerer on DD miss per J! rules
        else:
            # 3. Standard clue: process all buzz attempts (rebounds allowed)
            wager_amount = clue_value
            for attempt in clue.attempts:
                player = attempt.speaker

                # Validate speaker against known contestants
                if self.valid_contestants and player not in self.valid_contestants:
                    self.unknown_speaker_warnings += 1
                    if self.unknown_speaker_warnings <= 5:  # Cap log noise
                        logger.warning(
                            "State machine: skipping unknown speaker",
                            speaker=player,
                            selection_order=clue.selection_order,
                            valid_contestants=sorted(self.valid_contestants),
                        )
                    continue

                self.scores.setdefault(player, 0)
                self.coryat_scores.setdefault(player, 0)

                if attempt.is_correct:
                    self.scores[player] += wager_amount
                    self.coryat_scores[player] += wager_amount
                    # Shift board control ONLY on correct response
                    if self.current_board_control_contestant != player:
                        logger.info(
                            "Board control shift",
                            old=self.current_board_control_contestant,
                            new=player,
                        )
                    self.current_board_control_contestant = player
                    break  # Only one person can be correct per clue
                else:
                    self.scores[player] -= wager_amount
                    self.coryat_scores[player] -= wager_amount

        # 4. Enforce Chronological Rigidity for Score Adjustments
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
                    "Applied anchored adjustment",
                    index=current_index,
                    contestant=adj.contestant,
                    delta=adj.points_adjusted,
                    reason=adj.reason,
                )
            else:
                remaining_adjustments.append(adj)

        self.pending_adjustments = remaining_adjustments

    def process_final_jep(self, fj: FinalJep) -> None:
        """Processes Final J! wagers and updates scores."""
        for wager in fj.wagers_and_responses:
            player = wager.contestant
            if self.valid_contestants and player not in self.valid_contestants:
                self.unknown_speaker_warnings += 1
                if self.unknown_speaker_warnings <= 5:
                    logger.warning(
                        "State machine: skipping unknown speaker in FJ",
                        speaker=player,
                        valid_contestants=sorted(self.valid_contestants),
                    )
                continue

            self.scores.setdefault(player, 0)
            if wager.is_correct:
                self.scores[player] += wager.wager
            else:
                self.scores[player] -= wager.wager
