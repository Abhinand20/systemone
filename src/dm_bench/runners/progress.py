"""Shared live terminal progress and periodic log reporter for evaluation runners."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import time
from typing import Any

from rich.console import Console
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskID,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)


class EvalProgressTracker:
    """Report live progress bar (TTY) or periodic milestone lines (non-TTY) for `bench eval`."""

    def __init__(
        self,
        *,
        experiment_id: str,
        run_id: str,
        unit_label: str,
        total_units: int,
        already_completed: int = 0,
        rows: Sequence[str] = (),
        concurrency: int = 4,
        max_cost_usd: float = 10.0,
        initial_spent_usd: float = 0.0,
        run_dir: Path | None = None,
        console: Console | None = None,
        enabled: bool = True,
        log_interval_seconds: float = 5.0,
    ) -> None:
        self.experiment_id = experiment_id
        self.run_id = run_id
        self.unit_label = unit_label
        self.total_units = max(0, int(total_units))
        self.already_completed = max(0, min(self.total_units, int(already_completed)))
        self.pending_units = max(0, self.total_units - self.already_completed)
        self.rows = list(rows)
        self.concurrency = max(1, int(concurrency))
        self.max_cost_usd = float(max_cost_usd)
        self.run_dir = run_dir
        self.console = console or Console()
        self.enabled = enabled
        self.log_interval_seconds = max(0.5, float(log_interval_seconds))

        self._completed_units = self.already_completed
        self._calls = 0
        self._errors = 0
        self._illegal = 0
        self._spent_usd = max(0.0, float(initial_spent_usd))
        self._budget_halt_notified = False
        self._started_at = time.monotonic()
        self._last_log_at = self._started_at
        self._last_logged_units = self._completed_units
        self._step_interval = max(1, self.pending_units // 10) if self.pending_units > 0 else 1

        self._progress: Progress | None = None
        self._task_id: TaskID | None = None

    def _rows_preview(self) -> str:
        if not self.rows:
            return "-"
        if len(self.rows) <= 4:
            return ",".join(self.rows)
        return f"{','.join(self.rows[:3])},+{len(self.rows) - 3} more"

    def _stats_text(self) -> str:
        parts = [f"spent=${self._spent_usd:.5f}"]
        if self.unit_label != "decisions":
            parts.append(f"calls={self._calls}")
        if self._illegal:
            parts.append(f"illegal={self._illegal}")
        if self._errors:
            parts.append(f"err={self._errors}")
        return " | ".join(parts)

    def _print(self, message: str) -> None:
        if not self.enabled:
            return
        if self._progress is not None:
            self._progress.console.print(message)
        else:
            self.console.print(message)

    def __enter__(self) -> EvalProgressTracker:
        if not self.enabled:
            return self

        self._started_at = time.monotonic()
        self._last_log_at = self._started_at
        resume_note = (
            f" (resumed {self.already_completed}/{self.total_units})"
            if self.already_completed > 0
            else ""
        )
        dir_note = f" | dir={self.run_dir}" if self.run_dir is not None else ""
        self.console.print(
            f"[bold cyan]{self.experiment_id}[/bold cyan] run_id=[bold]{self.run_id}[/bold] | "
            f"pending={self.pending_units}/{self.total_units} {self.unit_label}{resume_note} | "
            f"rows={self._rows_preview()} | concurrency={self.concurrency} | "
            f"cap=${self.max_cost_usd:.2f}{dir_note}"
        )

        if self.pending_units > 0 and self.console.is_terminal:
            self._progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold cyan]{task.description}"),
                BarColumn(bar_width=28),
                MofNCompleteColumn(),
                TextColumn("{task.fields[unit]}"),
                TimeElapsedColumn(),
                TextColumn("eta"),
                TimeRemainingColumn(),
                TextColumn("{task.fields[stats]}"),
                console=self.console,
                transient=False,
            )
            self._progress.start()
            self._task_id = self._progress.add_task(
                self.experiment_id,
                total=self.total_units,
                completed=self.already_completed,
                unit=self.unit_label,
                stats=self._stats_text(),
            )
        return self

    def __exit__(self, exc_type: Any, exc: Any, tb: Any) -> None:
        if not self.enabled:
            return
        if self._progress is not None:
            if self._task_id is not None:
                self._progress.update(
                    self._task_id,
                    completed=self._completed_units,
                    stats=self._stats_text(),
                )
            self._progress.stop()
            self._progress = None
            self._task_id = None
        elif self.pending_units > 0 and self._completed_units != self._last_logged_units:
            self._log_milestone()

    def _log_milestone(self) -> None:
        pct = (100.0 * self._completed_units / self.total_units) if self.total_units > 0 else 100.0
        elapsed = time.monotonic() - self._started_at
        self._last_log_at = time.monotonic()
        self._last_logged_units = self._completed_units
        self.console.print(
            f"[{self.experiment_id}] {self._completed_units}/{self.total_units} {self.unit_label} "
            f"({pct:.0f}%) | {self._stats_text()} | elapsed={elapsed:.1f}s"
        )

    def record_decision(
        self,
        *,
        row_id: str,
        valid: bool,
        legal: bool,
        cost_usd: float,
        error_type: str | None = None,
        error_message: str | None = None,
        episode_id: str | None = None,
    ) -> None:
        """Update counters after one model call and emit a warning if the response failed."""
        self._calls += 1
        self._spent_usd += max(0.0, float(cost_usd or 0.0))
        if not valid:
            self._errors += 1
            err_kind = error_type or "Error"
            err_msg = (error_message or "").strip().splitlines()[0][:160] if error_message else ""
            loc = f" ({episode_id})" if episode_id else ""
            self._print(
                f"[yellow][{self.experiment_id}] {row_id}{loc} invalid response: "
                f"{err_kind}{': ' + err_msg if err_msg else ''}[/yellow]"
            )
        elif not legal:
            self._illegal += 1

        if self._progress is not None and self._task_id is not None:
            self._progress.update(self._task_id, stats=self._stats_text())
        elif (
            self.enabled
            and not self.console.is_terminal
            and self.pending_units > 0
            and (time.monotonic() - self._last_log_at) >= self.log_interval_seconds
        ):
            self._log_milestone()

    def advance(self, units: int = 1) -> None:
        """Advance completed work units (`decisions` in offline runs, `games` in T6)."""
        self._completed_units = min(self.total_units, self._completed_units + max(0, int(units)))
        if self._progress is not None and self._task_id is not None:
            self._progress.update(
                self._task_id,
                completed=self._completed_units,
                stats=self._stats_text(),
            )
            return

        if not self.enabled or self.pending_units == 0:
            return
        done_since_log = self._completed_units - self._last_logged_units
        elapsed_since_log = time.monotonic() - self._last_log_at
        if (
            self._completed_units == self.total_units
            or done_since_log >= self._step_interval
            or elapsed_since_log >= self.log_interval_seconds
        ):
            self._log_milestone()

    def note_budget_halt(self) -> None:
        """Emit a single notice when execution stops due to `--max-cost-usd`."""
        if self._budget_halt_notified:
            return
        self._budget_halt_notified = True
        self._print(
            f"[yellow][{self.experiment_id}] halted at budget cap "
            f"(${self._spent_usd:.6f} / ${self.max_cost_usd:.2f})[/yellow]"
        )
