"""Authoritative copied, already-cleaned inputs for the independent DCSA project."""
from __future__ import annotations

from hashlib import sha256
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED = PROJECT_ROOT / "data" / "processed"
CURVES_PATH = PROCESSED / "starrydata2_curves_long_clean.parquet"
SAMPLES_PATH = PROCESSED / "starrydata2_samples_clean.csv"


def input_paths() -> dict[str, Path]:
    return {"curves": CURVES_PATH, "samples": SAMPLES_PATH}


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_inputs() -> dict[str, dict[str, object]]:
    metadata = {}
    for name, path in input_paths().items():
        if not path.is_file():
            raise FileNotFoundError(path)
        metadata[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        }
    return metadata
