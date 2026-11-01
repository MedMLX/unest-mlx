"""Byte comparisons against locally available pinned Git source objects."""

from __future__ import annotations

import subprocess
from collections.abc import Mapping
from pathlib import Path


def require_git_sources(
    root: Path,
    revision: str,
    files: Mapping[str, str],
    repository: Path | None = None,
) -> None:
    """Require each consumed source file to equal its declared revision's Git blob.

    ``files`` maps snapshot-relative paths to paths in the verification repository.
    Archive/Hugging Face copies need ``repository`` pointing to a local Git object
    store. No fetch occurs, and unrelated working-tree changes are permitted.
    """
    proof = repository if repository is not None else root
    for relative, git_path in files.items():
        path = root / relative
        if not path.is_file():
            raise RuntimeError(f"Missing pinned source file: {path}")
        try:
            result = subprocess.run(
                (
                    "git",
                    "--no-lazy-fetch",
                    "--no-replace-objects",
                    "-C",
                    str(proof),
                    "show",
                    f"{revision}:{git_path}",
                ),
                capture_output=True,
                check=False,
            )
        except FileNotFoundError as error:
            raise RuntimeError("Source verification requires Git on PATH") from error
        if result.returncode:
            raise RuntimeError(
                f"Cannot verify {relative} at {revision}. Supply --source-repository "
                "with a local Git repository containing that revision and path; "
                "archive/Hugging Face directory names do not prove source identity. "
                f"Git reported: {result.stderr.decode(errors='replace').strip()}"
            )
        if path.read_bytes() != result.stdout:
            raise RuntimeError(f"Source differs from pinned revision {revision}: {path}")
