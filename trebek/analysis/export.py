"""
Multi-format episode export — exports structured game data to JSON, CSV, or Markdown.
"""

import csv
import io
import json
import os
import sqlite3
from typing import Any, Dict, List, Optional
from pathlib import Path


def _fetch_episode_data(db_path: str, episode_id: str) -> Optional[Dict[str, Any]]:
    """Gathers raw relational data for an episode."""
    if not os.path.exists(db_path):
        return None

    with sqlite3.connect(db_path) as conn:
        conn.row_factory = sqlite3.Row

        ep_row = conn.execute("SELECT * FROM episodes WHERE episode_id = ?", (episode_id,)).fetchone()
        if not ep_row:
            # Fallback to pipeline_state if not committed yet
            ps_row = conn.execute("SELECT * FROM pipeline_state WHERE episode_id = ?", (episode_id,)).fetchone()
            if not ps_row:
                return None
            return {"episode_id": episode_id, "status": ps_row["status"], "clues": [], "contestants": []}

        # Contestants & performances
        perf_rows = conn.execute(
            """
            SELECT c.contestant_id, c.name, c.occupational_category, c.is_returning_champion,
                   p.podium_position, p.coryat_score, p.final_score, p.forrest_bounce_index
            FROM episode_performances p
            JOIN contestants c ON p.contestant_id = c.contestant_id
            WHERE p.episode_id = ?
            ORDER BY p.podium_position ASC
            """,
            (episode_id,),
        ).fetchall()

        # Clues with wagers
        clue_rows = conn.execute(
            """
            SELECT c.*, w.actual_wager, w.running_score_at_time
            FROM clues c
            LEFT JOIN wagers w ON c.clue_id = w.clue_id
            WHERE c.episode_id = ?
            ORDER BY c.round, c.board_col, c.board_row, c.selection_order
            """,
            (episode_id,),
        ).fetchall()

        # Buzz attempts
        buzz_rows = conn.execute(
            """
            SELECT b.*, ct.name as contestant_name
            FROM buzz_attempts b
            JOIN clues cl ON b.clue_id = cl.clue_id
            JOIN contestants ct ON b.contestant_id = ct.contestant_id
            WHERE cl.episode_id = ?
            ORDER BY b.attempt_order ASC
            """,
            (episode_id,),
        ).fetchall()

        buzz_by_clue: Dict[str, List[Dict[str, Any]]] = {}
        for b in buzz_rows:
            cid = b["clue_id"]
            if cid not in buzz_by_clue:
                buzz_by_clue[cid] = []
            buzz_by_clue[cid].append(dict(b))

        clues = []
        for c in clue_rows:
            cd = dict(c)
            cd["buzz_attempts"] = buzz_by_clue.get(c["clue_id"], [])
            clues.append(cd)

        return {
            "episode_id": ep_row["episode_id"],
            "air_date": ep_row["air_date"],
            "host_name": ep_row["host_name"],
            "is_tournament": bool(ep_row["is_tournament"]),
            "contestants": [dict(p) for p in perf_rows],
            "clues": clues,
        }


def export_episode_json(db_path: str, episode_id: str, output_dir: Optional[str] = None) -> str:
    """Exports episode data as JSON. Prefers cached intermediate JSON if available."""
    if output_dir:
        json_path = Path(output_dir) / f"episode_{episode_id}.json"
        if json_path.exists():
            return json_path.read_text(encoding="utf-8")

    data = _fetch_episode_data(db_path, episode_id)
    if not data:
        raise ValueError(f"Episode {episode_id} not found in database.")
    return json.dumps(data, indent=2, default=str)


def export_episode_csv(db_path: str, episode_id: str) -> str:
    """Exports all clues and buzzes for an episode as CSV."""
    data = _fetch_episode_data(db_path, episode_id)
    if not data:
        raise ValueError(f"Episode {episode_id} not found in database.")

    output = io.StringIO()
    writer = csv.writer(output)

    writer.writerow(
        [
            "episode_id",
            "round",
            "category",
            "board_row",
            "board_col",
            "selection_order",
            "clue_text",
            "correct_response",
            "is_daily_double",
            "is_triple_stumper",
            "daily_double_wager",
            "wagerer_name",
            "first_buzz_contestant",
            "first_buzz_correct",
            "response_given",
        ]
    )

    for clue in data.get("clues", []):
        buzzes = clue.get("buzz_attempts", [])
        first_buzz = buzzes[0] if buzzes else {}
        writer.writerow(
            [
                episode_id,
                clue.get("round"),
                clue.get("category"),
                clue.get("board_row"),
                clue.get("board_col"),
                clue.get("selection_order"),
                clue.get("clue_text"),
                clue.get("correct_response"),
                clue.get("is_daily_double"),
                clue.get("is_triple_stumper"),
                clue.get("daily_double_wager") or clue.get("actual_wager"),
                clue.get("wagerer_name"),
                first_buzz.get("contestant_name", ""),
                first_buzz.get("is_correct", ""),
                first_buzz.get("response_given", ""),
            ]
        )

    return output.getvalue()


def export_episode_markdown(db_path: str, episode_id: str) -> str:
    """Exports episode data as a richly formatted Markdown dossier."""
    data = _fetch_episode_data(db_path, episode_id)
    if not data:
        raise ValueError(f"Episode {episode_id} not found in database.")

    lines = []
    lines.append(f"# Jeopardy! Episode {data['episode_id']}")
    lines.append("")

    meta_items = []
    if data.get("air_date"):
        meta_items.append(f"**Air Date:** {data['air_date']}")
    if data.get("host_name"):
        meta_items.append(f"**Host:** {data['host_name']}")
    if data.get("is_tournament"):
        meta_items.append("**Format:** Tournament Special")
    if meta_items:
        lines.append(" • ".join(meta_items))
        lines.append("")

    # Contestants table
    contestants = data.get("contestants", [])
    if contestants:
        lines.append("## Contestants & Coryat Scores")
        lines.append("")
        lines.append("| Podium | Name | Occupation | Coryat Score | Final Score |")
        lines.append("|:------:|:-----|:-----------|:------------:|:-----------:|")
        for c in contestants:
            coryat = f"${c.get('coryat_score'):,}" if c.get("coryat_score") is not None else "-"
            final = f"${c.get('final_score'):,}" if c.get("final_score") is not None else "-"
            lines.append(
                f"| {c.get('podium_position') or '-'} | {c.get('name', 'Unknown')} | {c.get('occupational_category') or '-'} | {coryat} | {final} |"
            )
        lines.append("")

    # Clues by round
    clues_by_round: Dict[str, List[Dict[str, Any]]] = {}
    for clue in data.get("clues", []):
        r = clue.get("round", "J!")
        if r not in clues_by_round:
            clues_by_round[r] = []
        clues_by_round[r].append(clue)

    for round_name in ("J!", "Double J!", "Final J!", "Tiebreaker"):
        round_clues = clues_by_round.get(round_name, [])
        if not round_clues:
            continue

        lines.append(f"## {round_name} Round")
        lines.append("")

        if round_name in ("J!", "Double J!"):
            # Group by category and order by row
            categories: Dict[str, List[Dict[str, Any]]] = {}
            for c in round_clues:
                cat = c.get("category", "Unknown")
                if cat not in categories:
                    categories[cat] = []
                categories[cat].append(c)

            lines.append("| Category | Row | Value | Clue | Correct Response | Status |")
            lines.append("|:---------|:---:|:-----:|:-----|:-----------------|:-------|")
            for cat, clist in categories.items():
                clist.sort(key=lambda x: (x.get("board_row") or 0, x.get("selection_order") or 0))
                for c in clist:
                    row_val = c.get("board_row") or "-"
                    base_val = (c.get("board_row") or 1) * (200 if round_name == "J!" else 400)
                    val_str = f"${base_val}"
                    if c.get("is_daily_double"):
                        val_str = f"**DD (${c.get('actual_wager') or c.get('daily_double_wager') or 'Wager'})**"
                    status_str = (
                        "Triple Stumper"
                        if c.get("is_triple_stumper")
                        else ("Verified" if c.get("is_verified") else "Solved")
                    )
                    clue_text = (c.get("clue_text") or "").replace("|", "\\|").replace("\n", " ")
                    resp = (c.get("correct_response") or "").replace("|", "\\|").replace("\n", " ")
                    lines.append(f"| {cat} | {row_val} | {val_str} | {clue_text} | {resp} | {status_str} |")
            lines.append("")
        else:
            # Final J!
            for c in round_clues:
                lines.append(f"**Category:** {c.get('category')}")
                lines.append(f"**Clue:** {c.get('clue_text')}")
                lines.append(f"**Correct Response:** {c.get('correct_response')}")
                lines.append("")

    return "\n".join(lines)


def export_episode(
    db_path: str,
    episode_id: str,
    format: str = "md",
    output_path: Optional[str] = None,
    output_dir: Optional[str] = None,
) -> str:
    """Dispatches export by format and optionally writes to output_path."""
    fmt = format.lower()
    if fmt == "json":
        content = export_episode_json(db_path, episode_id, output_dir=output_dir)
    elif fmt == "csv":
        content = export_episode_csv(db_path, episode_id)
    elif fmt in ("md", "markdown"):
        content = export_episode_markdown(db_path, episode_id)
    else:
        raise ValueError(f"Unsupported export format '{format}'. Use 'json', 'csv', or 'md'.")

    if output_path:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(content, encoding="utf-8")

    return content
