"""Sample images from OpenRL/DeepFakeFace without downloading the full 5 GB of zips.

DeepFakeFace mirrors IMDB-WIKI's layout: <source>/<NN>/<id>_<dates>.jpg, where NN is the last two
digits of the person's id. The same NN folder therefore holds the same people in every source,
which gives a clean identity split: test folders are never used for training.

Members are read straight out of the remote zips with HTTP range requests, in offset order so
reads stay sequential.

Usage:
    python training/download_dff_sample.py --split test   # folders 00-03, all four sources
    python training/download_dff_sample.py --split train  # folders 10-29, real + two fake sources
"""

import argparse
import time
import zipfile
from pathlib import Path

from huggingface_hub import HfFileSystem

ROOT = Path(__file__).resolve().parents[1]
REPO = "datasets/OpenRL/DeepFakeFace"

# Test: identity folders 00-03 from every source. text2img is never trained on.
# Train: identity folders 10-29. Real photos from all of them; each fake source from half,
# so real and fake are balanced and inpainted images are not all paired with their originals.
SPLITS = {
    "test": {
        "wiki": [f"{i:02d}" for i in range(0, 4)],
        "inpainting": [f"{i:02d}" for i in range(0, 4)],
        "insight": [f"{i:02d}" for i in range(0, 4)],
        "text2img": [f"{i:02d}" for i in range(0, 4)],
    },
    "train": {
        "wiki": [f"{i:02d}" for i in range(10, 30)],
        "inpainting": [f"{i:02d}" for i in range(10, 20)],
        "insight": [f"{i:02d}" for i in range(20, 30)],
    },
}
LIMITS = {"test": 1_000, "train": None}


def sample_source(fs: HfFileSystem, source: str, folders: set[str], out_dir: Path, limit: int | None) -> int:
    with fs.open(f"{REPO}/{source}.zip", "rb", block_size=4 << 20) as fh:
        archive = zipfile.ZipFile(fh)
        members = [
            info for info in archive.infolist()
            if info.filename.lower().endswith((".jpg", ".png"))
            and len(info.filename.split("/")) >= 3
            and info.filename.split("/")[-2] in folders
        ]
        members.sort(key=lambda info: info.header_offset)
        members = members[:limit]
        for info in members:
            dest = out_dir / source / info.filename.split("/", 1)[1]
            if dest.exists():
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(archive.read(info))
    return len(members)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--split", choices=SPLITS, required=True)
    args = parser.parse_args()

    out_dir = ROOT / ".hf_cache" / ("dff_sample" if args.split == "test" else "dff_train")
    fs = HfFileSystem()
    for source, folders in SPLITS[args.split].items():
        started = time.time()
        count = sample_source(fs, source, set(folders), out_dir, LIMITS[args.split])
        size = sum(p.stat().st_size for p in (out_dir / source).rglob("*.jpg")) / 1e6
        print(f"{source:11} {count:5} images  {size:6.0f} MB  {time.time() - started:4.0f}s", flush=True)


if __name__ == "__main__":
    main()
