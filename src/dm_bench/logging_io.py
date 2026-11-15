"""SHA-256 helpers, content-addressed PNG MediaStore, and JSONL/YAML RunLogger."""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Literal

import yaml
from pydantic import BaseModel

from dm_bench.config import find_repo_root
from dm_bench.contracts import (
    DecisionLogRow,
    DecisionResumeKey,
    EpisodeLogRow,
    EpisodeResumeKey,
    ImageFrame,
    RunManifest,
)


def sha256_bytes(data: bytes) -> str:
    """Return lowercase hexadecimal SHA-256 digest for raw bytes."""
    return hashlib.sha256(data).hexdigest()


def sha256_text(text: str) -> str:
    """Return lowercase hexadecimal SHA-256 digest for UTF-8 text."""
    return sha256_bytes(text.encode("utf-8"))


def sha256_json(obj: Any) -> str:
    """Return deterministic SHA-256 digest for any JSON-serializable structure."""
    canonical = json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, default=str)
    return sha256_text(canonical)


def utc_now_iso() -> str:
    """Return current UTC timestamp in ISO-8601 format with trailing Z."""
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def generate_run_id(prefix: str | None = None) -> str:
    """Generate a unique timestamped run identifier."""
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    suffix = uuid.uuid4().hex[:8]
    return f"{prefix}_{stamp}_{suffix}" if prefix else f"{stamp}_{suffix}"


def _atomic_write_bytes(target: Path, data: bytes) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(prefix=f".{target.name}.", dir=str(target.parent))
    tmp_path = Path(tmp_name)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_path, target)
    except Exception:
        if tmp_path.exists():
            tmp_path.unlink(missing_ok=True)
        raise


def _atomic_write_text(target: Path, text: str) -> None:
    _atomic_write_bytes(target, text.encode("utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    """Read a JSONL file into a list of dicts."""
    if not path.is_file():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            rows.append(json.loads(stripped))
    return rows


def write_jsonl(path: Path, records: Iterable[dict[str, Any] | BaseModel]) -> None:
    """Write records atomically as newline-delimited JSON."""
    lines: list[str] = []
    for rec in records:
        if isinstance(rec, BaseModel):
            lines.append(rec.model_dump_json())
        else:
            lines.append(json.dumps(rec, ensure_ascii=False))
    payload = ("\n".join(lines) + "\n") if lines else ""
    _atomic_write_text(path, payload)


class MediaStore:
    """Content-addressed PNG store saving frames at `experiments/media/frames/<sha256>.png`."""

    def __init__(
        self,
        media_root: Path | str | None = None,
        repo_root: Path | None = None,
    ) -> None:
        root = Path(repo_root) if repo_root is not None else find_repo_root()
        self.repo_root = root
        self.media_root = Path(media_root) if media_root is not None else (root / "experiments" / "media")
        self.frames_dir = self.media_root / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()

    def frame_path(self, digest: str) -> Path:
        """Return absolute filesystem path for a frame digest."""
        return self.frames_dir / f"{digest}.png"

    def relative_frame_path(self, digest: str) -> str:
        """Return repo-relative (or media-relative) POSIX path string for logging."""
        path = self.frame_path(digest)
        try:
            return path.resolve().relative_to(self.repo_root.resolve()).as_posix()
        except ValueError:
            return path.as_posix()

    def exists(self, digest: str) -> bool:
        """Check whether a frame with `digest` already exists on disk."""
        return self.frame_path(digest).is_file()

    def save_png(self, png_bytes: bytes) -> tuple[str, Path]:
        """Persist PNG bytes under `<media_root>/frames/<sha256>.png` and return `(sha256, path)`."""
        digest = sha256_bytes(png_bytes)
        target = self.frame_path(digest)
        with self._lock:
            if not target.is_file():
                _atomic_write_bytes(target, png_bytes)
        return digest, target

    def save_frame(self, frame: ImageFrame) -> ImageFrame:
        """Persist an `ImageFrame` and return a copy populated with `sha256` and `path`."""
        digest, _ = self.save_png(frame.png_bytes)
        rel_path = self.relative_frame_path(digest)
        return frame.model_copy(update={"sha256": digest, "path": rel_path})

    def load_png(self, digest: str) -> bytes:
        """Load raw PNG bytes for a stored SHA-256 digest."""
        return self.frame_path(digest).read_bytes()


class RunLogger:
    """Thread-safe streaming JSONL logger and manifest manager with checkpoint resume."""

    def __init__(self, run_dir: Path | str, manifest: RunManifest | None = None) -> None:
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.decisions_path = self.run_dir / "decisions.jsonl"
        self.episodes_path = self.run_dir / "episodes.jsonl"
        self.manifest_path = self.run_dir / "manifest.yaml"
        self._lock = threading.Lock()
        self._manifest = manifest
        if manifest is not None and not self.manifest_path.exists():
            self.write_manifest(manifest)

    def write_manifest(self, manifest: RunManifest | None = None) -> Path:
        """Atomically write `manifest.yaml`."""
        with self._lock:
            target_manifest = manifest or self._manifest
            if target_manifest is None:
                raise ValueError("No RunManifest available to write")
            target_manifest = target_manifest.model_copy(update={"updated_at_utc": utc_now_iso()})
            self._manifest = target_manifest
            serialized = yaml.safe_dump(
                target_manifest.model_dump(mode="json"),
                sort_keys=False,
                allow_unicode=True,
            )
            _atomic_write_text(self.manifest_path, serialized)
            return self.manifest_path

    def load_manifest(self) -> RunManifest | None:
        """Load `manifest.yaml` if present."""
        if not self.manifest_path.is_file():
            return None
        with self.manifest_path.open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle)
        if not isinstance(data, dict):
            return None
        manifest = RunManifest.model_validate(data)
        with self._lock:
            self._manifest = manifest
        return manifest

    def log_decision(self, row: DecisionLogRow) -> None:
        """Append a single `DecisionLogRow` as one JSON line and flush immediately."""
        line = row.model_dump_json() + "\n"
        with self._lock:
            with self.decisions_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def log_episode(self, row: EpisodeLogRow) -> None:
        """Append a single `EpisodeLogRow` as one JSON line and flush immediately."""
        line = row.model_dump_json() + "\n"
        with self._lock:
            with self.episodes_path.open("a", encoding="utf-8") as handle:
                handle.write(line)
                handle.flush()

    def load_decisions(self) -> list[DecisionLogRow]:
        """Load all logged `DecisionLogRow` entries from `decisions.jsonl`."""
        if not self.decisions_path.is_file():
            return []
        rows: list[DecisionLogRow] = []
        with self._lock:
            with self.decisions_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    rows.append(DecisionLogRow.model_validate_json(stripped))
        return rows

    def load_episodes(self) -> list[EpisodeLogRow]:
        """Load all logged `EpisodeLogRow` entries from `episodes.jsonl`."""
        if not self.episodes_path.is_file():
            return []
        rows: list[EpisodeLogRow] = []
        with self._lock:
            with self.episodes_path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    stripped = line.strip()
                    if not stripped:
                        continue
                    rows.append(EpisodeLogRow.model_validate_json(stripped))
        return rows

    def load_existing_keys(self) -> set[DecisionResumeKey]:
        """Return set of `(row_id, episode_id, render_variant, text_format, action_mode)` keys."""
        return {row.resume_key() for row in self.load_decisions()}

    def load_existing_episode_keys(self) -> set[EpisodeResumeKey]:
        """Return set of `(row_id, episode_id, opponent_type, action_mode)` keys for T6 resume."""
        return {row.resume_key() for row in self.load_episodes()}

    def finalize(
        self,
        status: Literal["completed", "halted_budget_cap", "failed"] = "completed",
        total_spent_usd: float | None = None,
        notes: str | None = None,
    ) -> RunManifest:
        """Recompute summary counts from JSONL logs, update manifest status, and persist."""
        decisions = self.load_decisions()
        episodes = self.load_episodes()
        manifest = self._manifest or self.load_manifest()
        if manifest is None:
            raise ValueError(f"Cannot finalize run without manifest in {self.run_dir}")

        now = utc_now_iso()
        spent = (
            float(total_spent_usd)
            if total_spent_usd is not None
            else sum(d.api_cost_usd for d in decisions)
        )
        valid_count = sum(1 for d in decisions if d.response_valid)
        illegal_count = sum(1 for d in decisions if d.response_valid and not d.action_is_legal)
        invalid_count = sum(1 for d in decisions if not d.response_valid)

        updated = manifest.model_copy(
            update={
                "status": status,
                "updated_at_utc": now,
                "completed_at_utc": now,
                "total_spent_usd": round(spent, 8),
                "total_decisions": len(decisions),
                "total_episodes": len(episodes),
                "valid_decisions": valid_count,
                "illegal_decisions": illegal_count,
                "invalid_decisions": invalid_count,
                "notes": notes if notes is not None else manifest.notes,
            }
        )
        self.write_manifest(updated)
        return updated
