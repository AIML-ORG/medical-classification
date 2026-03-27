import io
from pathlib import Path

import polars as pl
from PIL import Image

splits = {'train': 'data/train-00000-of-00001.parquet', 'validation': 'data/validation-00000-of-00001.parquet', 'test': 'data/test-00000-of-00001.parquet'}
df = pl.read_parquet('hf://datasets/Technoculture/medical-prescriptions/' + splits['test'])

out_dir = Path('test_images')
out_dir.mkdir(parents=True, exist_ok=True)

# HF Image column is a struct {bytes, path}; Polars exposes each row as a dict.
images = df.select(pl.col('image')).to_series().to_list()
for i, raw in enumerate(images):
    if isinstance(raw, dict):
        blob = raw.get('bytes')
        path_str = raw.get('path')
        if blob is not None:
            pil_img = Image.open(io.BytesIO(blob))
            stem = Path(path_str).name if path_str else f'image_{i:06d}.png'
        elif path_str:
            pil_img = Image.open(path_str)
            stem = Path(path_str).name
        else:
            print(f'Skip row {i}: image has no bytes or path')
            continue
    else:
        pil_img = Image.open(raw)
        stem = Path(getattr(pil_img, 'filename', '') or f'image_{i:06d}.png').name

    out_path = out_dir / f'{i:06d}_{stem}'
    pil_img.save(out_path)

print(f'Saved {len(images)} image(s) to {out_dir.resolve()}')
