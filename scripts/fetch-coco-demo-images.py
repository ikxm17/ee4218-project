#!/usr/bin/env python3
"""Fetch a balanced set of COCO val2017 images for the TinyissimoYOLO demo.

The TinyissimoYOLO model used in this project was retrained on three COCO
classes: chair (62), bowl (51), cup (47). This script picks ~30 val2017
images that contain at least one of those categories so the demo GUI on the
Kria board has guaranteed non-empty detections.

The script is intentionally dependency-light: stdlib only (urllib, json,
zipfile, argparse). Image validity is checked downstream by the inference
pipeline; this script just downloads bytes.

Reruns are idempotent: cached annotations are reused, and images that
already exist on disk are skipped.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import urllib.error
import urllib.request
import zipfile
from pathlib import Path
from typing import Iterable

# COCO category id -> friendly name. These are the three classes the
# accelerator was retrained on.
TARGET_CATEGORIES: dict[int, str] = {
    62: "chair",
    51: "bowl",
    47: "cup",
}

ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)
ANNOTATIONS_MEMBER = "annotations/instances_val2017.json"
IMAGES_BASE_URL = "http://images.cocodataset.org/val2017/"

CACHE_DIR = Path(os.path.expanduser("~/.cache/coco-demo"))
ANNOTATIONS_CACHE = CACHE_DIR / "annotations" / "instances_val2017.json"

# The canonical known-good regression image the HDL accelerator has been
# verified against throughout development. Copied into the demo dir under
# REFERENCE_IMAGE_DST_NAME so it's selectable alongside the scraped COCO
# images without colliding with the 000000XXXXXX.jpg naming.
REFERENCE_IMAGE_SRC = Path("software/inference/data/input_image.jpg")
REFERENCE_IMAGE_DST_NAME = "reference_image.jpg"

# Min bbox area (in the original COCO image's pixel space) used to prefer
# images whose target object is large-ish. Smaller instances tend to become
# tiny boxes after the 256x256 resize the accelerator consumes, which
# makes detections unreliable.
MIN_AREA = 2000.0


def _download(url: str, dest: Path, *, chunk: int = 1 << 16) -> None:
    """Stream a URL to ``dest``. Retries once on transient failure."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    last_err: Exception | None = None
    for attempt in (1, 2):
        try:
            req = urllib.request.Request(
                url, headers={"User-Agent": "ee4218-coco-demo/1.0"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp, tmp.open("wb") as fh:
                while True:
                    buf = resp.read(chunk)
                    if not buf:
                        break
                    fh.write(buf)
            tmp.replace(dest)
            return
        except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
            last_err = exc
            if tmp.exists():
                tmp.unlink()
            if attempt == 1:
                time.sleep(1.5)
                continue
            raise RuntimeError(f"failed to download {url}: {exc}") from exc
    # Defensive: should be unreachable.
    raise RuntimeError(f"failed to download {url}: {last_err}")


def ensure_annotations() -> Path:
    """Make sure ``instances_val2017.json`` is in the cache; return its path."""
    if ANNOTATIONS_CACHE.is_file():
        return ANNOTATIONS_CACHE

    print(f"[anno] downloading {ANNOTATIONS_URL} (~240 MB) ...", flush=True)
    zip_path = CACHE_DIR / "annotations_trainval2017.zip"
    _download(ANNOTATIONS_URL, zip_path)

    print(f"[anno] extracting {ANNOTATIONS_MEMBER} ...", flush=True)
    ANNOTATIONS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path) as zf:
        with zf.open(ANNOTATIONS_MEMBER) as src, ANNOTATIONS_CACHE.open("wb") as dst:
            shutil.copyfileobj(src, dst)

    # Don't keep the 240 MB zip lying around; the JSON is enough.
    try:
        zip_path.unlink()
    except OSError:
        pass

    return ANNOTATIONS_CACHE


def load_annotations(path: Path) -> dict:
    print(f"[anno] loading {path} ...", flush=True)
    with path.open("r") as fh:
        return json.load(fh)


def pick_images(
    coco: dict, per_class: int
) -> tuple[list[tuple[int, dict]], dict[int, list[int]]]:
    """Return ``(picks, per_class_image_ids)``.

    ``picks`` is a deduped list of ``(image_id, image_record)`` tuples in
    insertion order. ``per_class_image_ids`` is the per-category image ids
    that *would have been* selected (used for the manifest, may overlap).
    """
    images_by_id = {img["id"]: img for img in coco["images"]}

    # Collect (area, image_id) for each target category.
    by_cat: dict[int, list[tuple[float, int]]] = {
        cid: [] for cid in TARGET_CATEGORIES
    }
    for ann in coco["annotations"]:
        cid = ann["category_id"]
        if cid in by_cat:
            by_cat[cid].append((float(ann.get("area", 0.0)), ann["image_id"]))

    per_class_ids: dict[int, list[int]] = {}
    picks: list[tuple[int, dict]] = []
    seen: set[int] = set()

    for cid, name in TARGET_CATEGORIES.items():
        # Sort largest-area instances first; prefer area > MIN_AREA but
        # fall back to anything if not enough big ones exist.
        entries = sorted(by_cat[cid], key=lambda t: t[0], reverse=True)
        big = [iid for area, iid in entries if area >= MIN_AREA]
        small = [iid for area, iid in entries if area < MIN_AREA]
        ordered = big + small

        chosen: list[int] = []
        chosen_dedup: list[int] = []
        for iid in ordered:
            if iid in chosen:
                continue
            chosen.append(iid)
            if iid not in seen:
                chosen_dedup.append(iid)
            if len(chosen) >= per_class:
                break

        per_class_ids[cid] = chosen
        for iid in chosen:
            if iid in seen:
                continue
            seen.add(iid)
            picks.append((iid, images_by_id[iid]))

        print(
            f"[pick] {name:>5} (cid={cid}): selected {len(chosen)} "
            f"({len(chosen_dedup)} new)",
            flush=True,
        )

    return picks, per_class_ids


def copy_reference_image(out_dir: Path) -> Path | None:
    """Copy the canonical regression image into the demo dir.

    Returns the destination path on success, ``None`` if the source file
    is missing (not a hard error — the scraper should still succeed even
    if the reference image isn't available locally). Idempotent: skips
    the copy if the destination already exists and is non-empty.
    """
    src = REFERENCE_IMAGE_SRC
    if not src.is_file():
        print(
            f"[ref] WARN: {src} not found, skipping reference image",
            file=sys.stderr,
            flush=True,
        )
        return None

    dst = out_dir / REFERENCE_IMAGE_DST_NAME
    if dst.exists() and dst.stat().st_size > 0:
        print(f"[ref] {dst.name} already present, skipping copy", flush=True)
        return dst

    out_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    print(f"[ref] {src} -> {dst} ({dst.stat().st_size} B)", flush=True)
    return dst


def download_images(
    picks: Iterable[tuple[int, dict]], out_dir: Path
) -> tuple[list[Path], list[tuple[str, str]]]:
    """Download images. Returns ``(succeeded_paths, failures)``."""
    out_dir.mkdir(parents=True, exist_ok=True)
    ok: list[Path] = []
    failures: list[tuple[str, str]] = []
    for iid, rec in picks:
        fname = rec["file_name"]
        dest = out_dir / fname
        if dest.exists() and dest.stat().st_size > 0:
            ok.append(dest)
            continue
        url = IMAGES_BASE_URL + fname
        try:
            _download(url, dest)
            ok.append(dest)
            print(f"[img] {fname} ({dest.stat().st_size} B)", flush=True)
        except Exception as exc:  # noqa: BLE001 - we want to keep going
            failures.append((fname, str(exc)))
            print(f"[img] FAILED {fname}: {exc}", file=sys.stderr, flush=True)
    return ok, failures


def write_manifest(
    out_dir: Path,
    picks: list[tuple[int, dict]],
    per_class_ids: dict[int, list[int]],
    succeeded: set[str],
    reference_path: Path | None = None,
) -> Path:
    manifest_path = out_dir / "_manifest.json"

    # Build per-image categories: a single image may legitimately appear
    # for multiple target categories (it can contain both a chair and a
    # cup, for example), so collect all matching categories.
    image_cats: dict[int, list[str]] = {}
    for cid, ids in per_class_ids.items():
        name = TARGET_CATEGORIES[cid]
        for iid in ids:
            image_cats.setdefault(iid, []).append(name)

    entries = []
    for iid, rec in picks:
        if rec["file_name"] not in succeeded:
            continue
        entries.append(
            {
                "image_id": iid,
                "file_name": rec["file_name"],
                "width": rec.get("width"),
                "height": rec.get("height"),
                "source": "coco_val2017",
                "categories": sorted(image_cats.get(iid, [])),
            }
        )

    # Tack the canonical regression image onto the manifest with a
    # distinct source tag so downstream consumers can tell it apart
    # from the COCO picks.
    if reference_path is not None and reference_path.exists():
        entries.append(
            {
                "image_id": None,
                "file_name": reference_path.name,
                "width": None,
                "height": None,
                "source": "local_reference",
                "origin": str(REFERENCE_IMAGE_SRC),
                "categories": [],
            }
        )

    payload = {
        "source": "COCO val2017",
        "categories": {str(cid): name for cid, name in TARGET_CATEGORIES.items()},
        "min_area_px2": MIN_AREA,
        "count": len(entries),
        "images": entries,
    }

    with manifest_path.open("w") as fh:
        json.dump(payload, fh, indent=2)
    return manifest_path


def summarize(
    out_dir: Path,
    succeeded: list[Path],
    per_class_ids: dict[int, list[int]],
    succeeded_names: set[str],
    picks: list[tuple[int, dict]],
) -> None:
    name_to_id = {rec["file_name"]: iid for iid, rec in picks}
    succeeded_iids = {name_to_id[p.name] for p in succeeded if p.name in name_to_id}

    total_bytes = sum(p.stat().st_size for p in succeeded)
    print()
    print("=" * 60)
    print(f"output dir : {out_dir}")
    print(f"files      : {len(succeeded)}")
    print(f"total size : {total_bytes / 1024:.1f} KiB")
    for cid, name in TARGET_CATEGORIES.items():
        n = sum(1 for iid in per_class_ids[cid] if iid in succeeded_iids)
        print(f"  {name:>5} (cid={cid}): {n}")
    print("=" * 60)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("software/models/demo_images"),
        help="output directory for downloaded JPEGs",
    )
    parser.add_argument(
        "--n",
        type=int,
        default=30,
        help="approximate target image count (default 30)",
    )
    parser.add_argument(
        "--per-class",
        type=int,
        default=10,
        help="images selected per target class before dedup (default 10)",
    )
    args = parser.parse_args(argv)

    # Allow callers to pass --n without --per-class and have things still
    # roughly balance out across the three classes.
    per_class = args.per_class
    if per_class * len(TARGET_CATEGORIES) < args.n:
        per_class = max(per_class, (args.n + len(TARGET_CATEGORIES) - 1) // len(TARGET_CATEGORIES))

    out_dir = args.out.resolve()
    print(f"[cfg] target dir = {out_dir}")
    print(f"[cfg] per-class  = {per_class}")
    print(f"[cfg] target n   = {args.n}")

    ensure_annotations()
    coco = load_annotations(ANNOTATIONS_CACHE)

    picks, per_class_ids = pick_images(coco, per_class)
    print(f"[pick] {len(picks)} unique images selected")

    succeeded, failures = download_images(picks, out_dir)
    succeeded_names = {p.name for p in succeeded}

    # Always also drop the canonical known-good regression image into
    # the demo dir. It's outside the COCO picks but useful as an
    # always-available sanity check in the demo GUI.
    reference_path = copy_reference_image(out_dir)
    if reference_path is not None:
        succeeded.append(reference_path)
        succeeded_names.add(reference_path.name)

    manifest_path = write_manifest(
        out_dir, picks, per_class_ids, succeeded_names,
        reference_path=reference_path,
    )
    print(f"[manifest] wrote {manifest_path}")

    summarize(out_dir, succeeded, per_class_ids, succeeded_names, picks)

    if failures:
        print(f"[warn] {len(failures)} downloads failed:", file=sys.stderr)
        for fname, err in failures:
            print(f"  - {fname}: {err}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
