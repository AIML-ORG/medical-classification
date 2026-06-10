"""
Find exact, visual (pHash), D4-canonical, and embedding-based near-duplicates
under a datasets root. Writes Polars Parquet outputs and optional JSON report POST.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import time
import urllib.error
import urllib.request
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import imagehash
import numpy as np
import polars as pl
from PIL import Image, ImageOps
from tqdm import tqdm

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# Default excludes aligned with v3/build_manifest.py intent (substring match on resolved path parts)
DEFAULT_EXCLUDE_SUBSTRINGS = frozenset(
    {
        "images_augmented",
    }
)

def _d4_transpose_ops() -> tuple:
    t = getattr(Image, "Transpose", None)
    if t is not None:
        return (
            t.FLIP_LEFT_RIGHT,
            t.FLIP_TOP_BOTTOM,
            t.ROTATE_90,
            t.ROTATE_180,
            t.ROTATE_270,
            t.TRANSPOSE,
            t.TRANSVERSE,
        )
    return (
        Image.FLIP_LEFT_RIGHT,
        Image.FLIP_TOP_BOTTOM,
        Image.ROTATE_90,
        Image.ROTATE_180,
        Image.ROTATE_270,
        Image.TRANSPOSE,
        Image.TRANSVERSE,
    )


D4_OPS: tuple = _d4_transpose_ops()


def repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def enumerate_images(
    root: Path,
    extensions: set[str],
    exclude_substrings: set[str],
    max_images: int | None,
) -> list[Path]:
    root = root.resolve()
    out: list[Path] = []
    for p in root.rglob("*"):
        if not p.is_file():
            continue
        if p.suffix.lower() not in extensions:
            continue
        resolved = str(p.resolve())
        if any(s in resolved for s in exclude_substrings):
            continue
        out.append(p)
        if max_images is not None and len(out) >= max_images:
            break
    out.sort(key=lambda x: str(x))
    return out


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def load_rgb_image(path: Path, max_side: int) -> Image.Image | None:
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im)
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
            return im.copy()
    except OSError as e:
        print(f"Skip (decode error) {path}: {e}")
        return None


def phash_of_pil(im: Image.Image) -> imagehash.ImageHash:
    return imagehash.phash(im)


def phash_prefix_bits(h: imagehash.ImageHash, prefix_bits: int) -> int:
    flat = h.hash.flatten()
    val = 0
    for i in range(prefix_bits):
        bit = bool(flat[i])
        val = (val << 1) | int(bit)
    return val


def canonical_d4_phash(im: Image.Image) -> tuple[str, imagehash.ImageHash]:
    hashes: list[imagehash.ImageHash] = [phash_of_pil(im)]
    for op in D4_OPS:
        try:
            t = im.transpose(op)
            hashes.append(phash_of_pil(t))
        except OSError:
            continue
    best = min(hashes, key=lambda x: str(x))
    return str(best), best


class UnionFind:
    def __init__(self, n: int) -> None:
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1


def cluster_indices_by_hamming(
    phashes: list[imagehash.ImageHash],
    prefix_bits: int,
    max_hamming: int,
) -> list[int]:
    n = len(phashes)
    uf = UnionFind(n)
    buckets: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(phashes):
        buckets[phash_prefix_bits(h, prefix_bits)].append(i)
    for ids in buckets.values():
        m = len(ids)
        for ii in range(m):
            i = ids[ii]
            for jj in range(ii + 1, m):
                j = ids[jj]
                if phashes[i] - phashes[j] <= max_hamming:
                    uf.union(i, j)
    roots = [uf.find(i) for i in range(n)]
    root_to_gid: dict[int, int] = {}
    gids: list[int] = []
    next_gid = 0
    for r in roots:
        if r not in root_to_gid:
            root_to_gid[r] = next_gid
            next_gid += 1
        gids.append(root_to_gid[r])
    return gids


def run_exact(paths: list[Path]) -> pl.DataFrame:
    rows: list[dict] = []
    for p in tqdm(paths, desc="SHA256"):
        try:
            digest = sha256_file(p)
            size_b = p.stat().st_size
        except OSError as e:
            print(f"Skip (read error) {p}: {e}")
            continue
        rows.append({"sha256": digest, "path": str(p.resolve()), "size_bytes": size_b})
    if not rows:
        return pl.DataFrame(
            schema={
                "group_id": pl.Int64,
                "sha256": pl.Utf8,
                "path": pl.Utf8,
                "size_bytes": pl.Int64,
            }
        )
    df = pl.DataFrame(rows)
    counts = df.group_by("sha256").len().filter(pl.col("len") > 1)
    dup_hashes = set(counts["sha256"].to_list())
    df = df.filter(pl.col("sha256").is_in(dup_hashes))
    df = df.sort(["sha256", "path"])
    df = df.with_columns(
        pl.col("sha256").rank(method="dense").cast(pl.Int64).alias("group_id")
    )
    return df.select(["group_id", "sha256", "path", "size_bytes"])


@dataclass
class PhashRow:
    path: str
    phash: imagehash.ImageHash


def compute_phash_rows(paths: list[Path], max_side: int) -> list[PhashRow]:
    rows: list[PhashRow] = []
    for p in tqdm(paths, desc="pHash decode"):
        im = load_rgb_image(p, max_side=max_side)
        if im is None:
            continue
        try:
            h = phash_of_pil(im)
        except OSError as e:
            print(f"Skip (phash error) {p}: {e}")
            continue
        rows.append(PhashRow(path=str(p.resolve()), phash=h))
    return rows


def build_visual_groups_parquet(
    rows: list[PhashRow],
    prefix_bits: int,
    max_hamming: int,
) -> pl.DataFrame:
    if len(rows) < 2:
        return pl.DataFrame(
            schema={
                "group_id": pl.Int64,
                "phash_hex": pl.Utf8,
                "path": pl.Utf8,
                "hamming_to_representative": pl.Int64,
            }
        )
    phashes = [r.phash for r in rows]
    paths = [r.path for r in rows]
    gids = cluster_indices_by_hamming(phashes, prefix_bits, max_hamming)
    by_gid: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(gids):
        by_gid[g].append(i)
    out_rows: list[dict] = []
    next_out_gid = 0
    for g, idxs in sorted(by_gid.items(), key=lambda x: x[0]):
        if len(idxs) < 2:
            continue
        rep_i = min(idxs, key=lambda i: paths[i])
        rep_hash = phashes[rep_i]
        rep_hex = str(rep_hash)
        for i in sorted(idxs, key=lambda j: paths[j]):
            d = phashes[i] - rep_hash
            out_rows.append(
                {
                    "group_id": next_out_gid,
                    "phash_hex": rep_hex,
                    "path": paths[i],
                    "hamming_to_representative": int(d),
                }
            )
        next_out_gid += 1
    if not out_rows:
        return pl.DataFrame(
            schema={
                "group_id": pl.Int64,
                "phash_hex": pl.Utf8,
                "path": pl.Utf8,
                "hamming_to_representative": pl.Int64,
            }
        )
    return pl.DataFrame(out_rows)


def build_d4_near_pairs(
    paths_for_phash: list[Path],
    max_side: int,
    prefix_bits: int,
    max_hamming: int,
) -> pl.DataFrame:
    d4_list: list[tuple[imagehash.ImageHash, str]] = []
    for p in tqdm(paths_for_phash, desc="D4 canonical pHash"):
        im = load_rgb_image(p, max_side=max_side)
        if im is None:
            continue
        try:
            _, h = canonical_d4_phash(im)
        except OSError as e:
            print(f"Skip (D4 phash error) {p}: {e}")
            continue
        d4_list.append((h, str(p.resolve())))
    if len(d4_list) < 2:
        return pl.DataFrame(
            schema={
                "path_a": pl.Utf8,
                "path_b": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    phashes = [t[0] for t in d4_list]
    paths = [t[1] for t in d4_list]
    gids = cluster_indices_by_hamming(phashes, prefix_bits, max_hamming)
    by_gid: dict[int, list[int]] = defaultdict(list)
    for i, g in enumerate(gids):
        by_gid[g].append(i)
    pair_rows: list[dict] = []
    max_bits = 64.0
    for _g, idxs in sorted(by_gid.items(), key=lambda x: x[0]):
        if len(idxs) < 2:
            continue
        rep_i = min(idxs, key=lambda i: paths[i])
        for j in idxs:
            if j == rep_i:
                continue
            d = phashes[j] - phashes[rep_i]
            sim = 1.0 - (float(d) / max_bits)
            pair_rows.append(
                {
                    "path_a": paths[rep_i],
                    "path_b": paths[j],
                    "similarity": sim,
                    "method": "d4_phash",
                }
            )
    if not pair_rows:
        return pl.DataFrame(
            schema={
                "path_a": pl.Utf8,
                "path_b": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    return pl.DataFrame(pair_rows)


def run_embedding_pairs(
    paths: list[Path],
    max_side: int,
    batch_size: int,
    embed_min_sim: float,
    nn_k: int,
    model_name: str,
) -> pl.DataFrame:
    import torch
    import timm
    from timm.data import create_transform, resolve_model_data_config

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Embedding device: {device}")
    model = timm.create_model(model_name, pretrained=True, num_classes=0)
    model = model.to(device)
    model.eval()
    data_config = resolve_model_data_config(model)
    transform = create_transform(**data_config, is_training=False)

    valid_paths: list[Path] = []
    feats_list: list[np.ndarray] = []
    for start in tqdm(range(0, len(paths), batch_size), desc="Embedding batches"):
        batch_paths = paths[start : start + batch_size]
        tensors: list[torch.Tensor] = []
        for p in batch_paths:
            im = load_rgb_image(p, max_side=max_side)
            if im is None:
                continue
            try:
                t = transform(im)
                tensors.append(t)
                valid_paths.append(p)
            except OSError as e:
                print(f"Skip (embed transform) {p}: {e}")
        if not tensors:
            continue
        batch = torch.stack(tensors, dim=0).to(device)
        with torch.no_grad():
            feat = model(batch)
            if feat.dim() > 2:
                feat = feat.mean(dim=(2, 3))
        feat = torch.nn.functional.normalize(feat.float(), dim=1)
        feats_list.append(feat.cpu().numpy())
    if not feats_list:
        return pl.DataFrame(
            schema={
                "path_a": pl.Utf8,
                "path_b": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    feats = np.vstack(feats_list)
    n = feats.shape[0]
    if n < 2:
        return pl.DataFrame(
            schema={
                "path_a": pl.Utf8,
                "path_b": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    path_strs = [str(p.resolve()) for p in valid_paths]

    from sklearn.neighbors import NearestNeighbors

    k = min(max(nn_k, 2), n)
    nn = NearestNeighbors(n_neighbors=k, metric="cosine", algorithm="brute")
    nn.fit(feats)
    dists, idxs = nn.kneighbors(feats, return_distance=True)
    pair_set: set[tuple[str, str]] = set()
    pair_rows: list[dict] = []
    for i in range(n):
        for dist, j in zip(dists[i], idxs[i]):
            if j == i:
                continue
            cos_sim = 1.0 - float(dist)
            if cos_sim < embed_min_sim:
                continue
            a, b = path_strs[i], path_strs[j]
            key = (a, b) if a < b else (b, a)
            if key in pair_set:
                continue
            pair_set.add(key)
            pair_rows.append(
                {
                    "path_a": key[0],
                    "path_b": key[1],
                    "similarity": cos_sim,
                    "method": "embedding",
                }
            )
    if not pair_rows:
        return pl.DataFrame(
            schema={
                "path_a": pl.Utf8,
                "path_b": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    return pl.DataFrame(pair_rows)


def sample_group_ids(df: pl.DataFrame, col: str, k: int = 5) -> list:
    if df.is_empty() or col not in df.columns:
        return []
    u = df[col].unique().to_list()
    if len(u) <= k:
        return u
    rng = random.Random(42)
    return rng.sample(u, k)


def maybe_post_report(
    url: str | None,
    payload: dict,
) -> None:
    if not url:
        return
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            print(f"DEDUP_REPORT_URL status: {resp.status}")
    except urllib.error.URLError as e:
        print(f"DEDUP_REPORT_URL failed: {e}")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Dataset duplicate and near-duplicate detection")
    p.add_argument(
        "--datasets-root",
        type=Path,
        default=None,
        help="Root folder to scan (default: <repo>/v3/datasets)",
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Parquet output directory (default: <repo>/v3/data/dedup)",
    )
    p.add_argument("--max-side", type=int, default=512, help="Max side for pHash / embed decode")
    p.add_argument("--phash-prefix-bits", type=int, default=16, help="Bucket prefix bits for Hamming clustering")
    p.add_argument(
        "--phash-max-hamming",
        type=int,
        default=6,
        help="Max Hamming distance for visual (raw pHash) clusters",
    )
    p.add_argument(
        "--d4-phash-max-hamming",
        type=int,
        default=None,
        help="Max Hamming for D4-canonical clusters (default: same as --phash-max-hamming)",
    )
    p.add_argument("--skip-exact", action="store_true")
    p.add_argument("--skip-phash", action="store_true")
    p.add_argument("--skip-d4", action="store_true")
    p.add_argument("--skip-embedding", action="store_true", help="Skip timm embedding near-duplicate pairs")
    p.add_argument(
        "--skip-near",
        action="store_true",
        help="Skip both D4 pHash and embedding near-duplicate tiers",
    )
    p.add_argument("--embed-min-sim", type=float, default=0.95, help="Min cosine similarity for embedding pairs")
    p.add_argument("--embed-batch-size", type=int, default=32)
    p.add_argument("--embed-nn-k", type=int, default=20, help="Neighbors per image to check for embedding tier")
    p.add_argument(
        "--embed-model",
        type=str,
        default="mobilenetv4_conv_small.e3600_r256_in1k",
        help="timm model name for embeddings",
    )
    p.add_argument("--max-images", type=int, default=None, help="Cap images for debugging")
    p.add_argument(
        "--exclude-substring",
        action="append",
        default=[],
        help="Exclude files whose resolved path contains this substring (repeatable)",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.skip_near:
        args.skip_d4 = True
        args.skip_embedding = True
    root = repo_root()
    datasets_root = (args.datasets_root or root / "v3" / "datasets").resolve()
    output_dir = (args.output_dir or root / "v3" / "data" / "dedup").resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    exclude = set(DEFAULT_EXCLUDE_SUBSTRINGS) | set(args.exclude_substring)

    print(f"Datasets root: {datasets_root}")
    print(f"Output dir: {output_dir}")
    print(f"Exclude substrings: {sorted(exclude)}")

    t0 = time.perf_counter()
    paths = enumerate_images(
        datasets_root,
        IMAGE_EXTENSIONS,
        exclude,
        args.max_images,
    )
    print(f"Found {len(paths)} image files")

    empty_near_schema = {
        "path_a": pl.Utf8,
        "path_b": pl.Utf8,
        "similarity": pl.Float64,
        "method": pl.Utf8,
    }
    empty_exact_schema = {
        "group_id": pl.Int64,
        "sha256": pl.Utf8,
        "path": pl.Utf8,
        "size_bytes": pl.Int64,
    }
    empty_visual_schema = {
        "group_id": pl.Int64,
        "phash_hex": pl.Utf8,
        "path": pl.Utf8,
        "hamming_to_representative": pl.Int64,
    }
    exact_df = pl.DataFrame(schema=empty_exact_schema)
    visual_df = pl.DataFrame(schema=empty_visual_schema)
    d4_df = pl.DataFrame(schema=empty_near_schema)
    embed_df = pl.DataFrame(schema=empty_near_schema)
    out_exact = output_dir / "exact_duplicate_groups.parquet"
    out_vis = output_dir / "visual_duplicate_groups.parquet"

    if not args.skip_exact:
        print("Tier 1: exact duplicates (SHA256)")
        exact_df = run_exact(paths)
        exact_df.write_parquet(out_exact)
        print(f"Wrote {out_exact} rows={len(exact_df)}")
    else:
        exact_df.write_parquet(out_exact)
        print(f"Skipped tier 1; wrote empty {out_exact}")

    if not args.skip_phash:
        print("Tier 2: visual clusters (raw pHash + Hamming)")
        phash_rows = compute_phash_rows(paths, max_side=args.max_side)
        visual_df = build_visual_groups_parquet(
            phash_rows,
            prefix_bits=args.phash_prefix_bits,
            max_hamming=args.phash_max_hamming,
        )
        visual_df.write_parquet(out_vis)
        print(f"Wrote {out_vis} rows={len(visual_df)}")
    else:
        visual_df.write_parquet(out_vis)
        print(f"Skipped tier 2; wrote empty {out_vis}")

    d4_max = args.d4_phash_max_hamming
    if d4_max is None:
        d4_max = args.phash_max_hamming
    if not args.skip_d4:
        print("Tier 3a: D4-canonical pHash near-duplicate pairs")
        d4_df = build_d4_near_pairs(
            paths,
            max_side=args.max_side,
            prefix_bits=args.phash_prefix_bits,
            max_hamming=d4_max,
        )

    if not args.skip_embedding:
        print("Tier 3b: embedding near-duplicate pairs")
        embed_df = run_embedding_pairs(
            paths,
            max_side=args.max_side,
            batch_size=args.embed_batch_size,
            embed_min_sim=args.embed_min_sim,
            nn_k=args.embed_nn_k,
            model_name=args.embed_model,
        )

    near_parts = [df for df in (d4_df, embed_df) if len(df) > 0]
    if near_parts:
        near_df = pl.concat(near_parts, how="vertical")
    else:
        near_df = pl.DataFrame(
            schema={
                "path_a": pl.Utf8,
                "path_b": pl.Utf8,
                "similarity": pl.Float64,
                "method": pl.Utf8,
            }
        )
    out_near = output_dir / "near_duplicate_pairs.parquet"
    near_df.write_parquet(out_near)
    print(f"Wrote {out_near} rows={len(near_df)}")

    elapsed = time.perf_counter() - t0
    print(f"Done in {elapsed:.1f}s")

    report_url = os.environ.get("DEDUP_REPORT_URL")
    payload = {
        "timestamp_unix": int(time.time()),
        "datasets_root": str(datasets_root),
        "output_dir": str(output_dir),
        "image_count": len(paths),
        "elapsed_seconds": round(elapsed, 3),
        "exact_duplicate_rows": len(exact_df),
        "visual_duplicate_rows": len(visual_df),
        "near_duplicate_rows": len(near_df),
        "samples": {
            "exact_group_id": sample_group_ids(exact_df, "group_id", 5),
            "visual_group_id": sample_group_ids(visual_df, "group_id", 5),
            "near_method": sample_group_ids(near_df, "method", 5),
        },
    }
    maybe_post_report(report_url, payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
