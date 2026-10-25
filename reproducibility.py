"""Small output-safety helpers for reproducible experiment runs."""
from pathlib import Path
from datetime import datetime, timezone


def require_fresh_output_dir(path: str | Path, *, allow_existing: tuple[str, ...] = ()) -> Path:
    """Refuse a new run in a directory that already contains prior artifacts."""
    output = Path(path)
    if output.exists() and not output.is_dir():
        raise FileExistsError(f"output path is not a directory: {output}")
    if output.exists():
        existing = [item for item in output.rglob("*") if item.is_file()
                    and item.name not in allow_existing]
        if existing:
            first = existing[0]
            raise FileExistsError(
                f"output directory already contains experiment artifacts ({first}); "
                "choose a new --output-dir to preserve prior results"
            )
    output.mkdir(parents=True, exist_ok=True)
    return output


def create_unique_run_dir(parent: str | Path, prefix: str) -> Path:
    """Create a timestamped run directory without reusing an existing path."""
    root = Path(parent)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    candidate = root / f"{prefix}_{stamp}"
    suffix = 1
    while candidate.exists():
        candidate = root / f"{prefix}_{stamp}_{suffix}"
        suffix += 1
    candidate.mkdir()
    return candidate


def require_outputs_absent(directory: str | Path, filenames: tuple[str, ...]) -> None:
    """Prevent an experiment writer from replacing artifacts from an earlier run."""
    root = Path(directory)
    existing = [root / name for name in filenames if (root / name).exists()]
    if existing:
        raise FileExistsError(
            f"experiment output already exists ({existing[0]}); "
            "choose a fresh output directory to preserve prior results"
        )
