from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

DEFAULT_DATASETS_DIR = Path(__file__).resolve().parent / "datasets"


def parse_roboflow_project_id(project_id: str) -> Optional[tuple[str, str]]:
    s = project_id.strip().strip("/")
    if "/" not in s:
        print("Expected workspace/project, e.g. tf-test/prescription-dpflz")
        return None
    workspace, slug = s.split("/", 1)
    if not workspace or not slug:
        print("Expected workspace/project, e.g. tf-test/prescription-dpflz")
        return None
    return workspace, slug


def download_roboflow_dataset(
    project_id: str,
    *,
    version: Optional[int] = None,
    datasets_dir: Optional[Path | str] = None,
    model_format: str = "yolov8",
    api_key: Optional[str] = None,
    api_key_env: str = "ROBOFLOW_API_KEY",
    overwrite: bool = False,
) -> Optional[Path]:
    parsed = parse_roboflow_project_id(project_id)
    if parsed is None:
        return None
    workspace, project_slug = parsed

    key = (api_key or os.environ.get(api_key_env, "").strip())
    if not key:
        print(f"Set {api_key_env} in Windows environment variables")
        return None

    from roboflow import Roboflow

    root = Path(datasets_dir) if datasets_dir is not None else DEFAULT_DATASETS_DIR
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)

    rf = Roboflow(api_key=key)
    proj = rf.workspace(workspace).project(project_slug)

    if version is None:
        infos = proj.get_version_information()
        if not infos:
            print("No dataset versions found")
            return None
        version = max(int(os.path.basename(v["id"])) for v in infos)

    target_dir = root / f"{workspace}_{project_slug}_v{version}"
    proj.version(version).download(model_format=model_format, location=str(target_dir), overwrite=overwrite)
    path = target_dir.resolve()
    print(path)
    return path


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("project_id", help="workspace/project, e.g. tf-test/prescription-dpflz")
    p.add_argument("--format", dest="model_format", default="yolov8")
    p.add_argument("--datasets-dir", type=Path, default=None)
    p.add_argument("--overwrite", action="store_true")
    a = p.parse_args()
    download_roboflow_dataset(
        a.project_id,
        datasets_dir=a.datasets_dir,
        model_format=a.model_format,
        overwrite=a.overwrite,
    )
