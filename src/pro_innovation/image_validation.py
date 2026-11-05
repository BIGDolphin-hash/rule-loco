"""Shared input-image validation for long-running memory construction."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Sequence

from PIL import Image


ProgressCallback = Callable[[str, int, int, str], None]


def filter_decodable_images(
    paths: Sequence[str | Path],
    *,
    progress_callback: ProgressCallback | None = None,
) -> tuple[Path, ...]:
    """Return fully decodable images and report corrupt files as skipped."""

    resolved = tuple(Path(path).expanduser().resolve() for path in paths)
    readable: list[Path] = []
    total = len(resolved)
    for index, path in enumerate(resolved, start=1):
        try:
            with Image.open(path) as image:
                image.load()
        # Pillow normally reports decode failures as OSError, but malformed
        # PNG chunks (for example a bad IDAT checksum) can raise SyntaxError
        # during image.load().  Both mean the input cannot be trusted as a
        # normal-memory reference and must be skipped.
        except (OSError, SyntaxError) as exc:
            if progress_callback is not None:
                progress_callback(
                    "skip-unreadable",
                    index,
                    total,
                    f"{path.name}: {exc}",
                )
            continue
        readable.append(path)
    return tuple(readable)
