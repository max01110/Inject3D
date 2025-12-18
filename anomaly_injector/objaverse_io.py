"""
Objaverse asset fetching with timeout handling.
"""
import os
import random
import signal
import zipfile
from pathlib import Path
from contextlib import contextmanager

import objaverse


class TimeoutError(RuntimeError):
    pass


@contextmanager
def time_limit(seconds):
    """Context manager that raises TimeoutError after given seconds (Unix only)."""
    if not seconds or seconds <= 0:
        yield
        return

    def handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old = signal.signal(signal.SIGALRM, handler)
    try:
        signal.alarm(int(seconds))
        yield
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old)


def _find_mesh_file(path):
    """Find supported mesh file at path (file, zip, or directory)."""
    exts = {".glb", ".gltf", ".obj", ".fbx"}
    p = Path(path)

    # Direct file
    if p.is_file() and p.suffix.lower() in exts:
        return str(p)

    # Zip archive
    if p.is_file() and p.suffix.lower() == ".zip":
        with zipfile.ZipFile(p, "r") as zf:
            for ext in exts:
                for name in zf.namelist():
                    if name.lower().endswith(ext):
                        out = p.parent / Path(name).name
                        if not out.exists():
                            zf.extract(name, p.parent)
                            (p.parent / name).rename(out)
                        return str(out)
        raise RuntimeError(f"No mesh in zip: {p}")

    # Directory
    if p.is_dir():
        for ext in exts:
            for cand in p.glob(f"*{ext}"):
                return str(cand)
        for ext in exts:
            for cand in p.rglob(f"*{ext}"):
                return str(cand)

    raise RuntimeError(f"No mesh found at: {path}")


def get_random_objaverse(download_dir, timeout_sec=60, retries=3):
    """
    Fetch random Objaverse asset with timeout and retries.
    Returns (uid, mesh_path).
    """
    os.makedirs(download_dir, exist_ok=True)
    last_err = None

    for attempt in range(1, retries + 1):
        try:
            with time_limit(timeout_sec):
                uids = list(objaverse.load_uids())
            if not uids:
                raise RuntimeError("No UIDs available")

            uid = random.choice(uids)

            # Try with download_dir kwarg first
            try:
                with time_limit(timeout_sec):
                    assets = objaverse.load_objects(uids=[uid], download_dir=download_dir)
            except TypeError:
                # Older API without download_dir
                cwd = os.getcwd()
                try:
                    os.chdir(download_dir)
                    with time_limit(timeout_sec):
                        assets = objaverse.load_objects(uids=[uid])
                finally:
                    os.chdir(cwd)

            if uid not in assets:
                raise RuntimeError(f"Asset not returned: {uid}")

            entry = assets[uid]
            if isinstance(entry, str):
                raw_path = entry
            elif isinstance(entry, dict):
                raw_path = (entry.get("file_path") or
                            (entry.get("paths", [None])[0]) or
                            next((v for v in entry.values()
                                  if isinstance(v, str) and os.path.exists(v)), None))
            else:
                raw_path = None

            if not raw_path:
                raise RuntimeError(f"Unknown asset format for {uid}")

            mesh_path = _find_mesh_file(raw_path)
            if not mesh_path or not os.path.exists(mesh_path):
                raise RuntimeError(f"Mesh not found: {raw_path}")

            return uid, mesh_path

        except TimeoutError as e:
            last_err = str(e)
            print(f"[Objaverse] Attempt {attempt}/{retries} timed out, retrying...")
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[Objaverse] Attempt {attempt}/{retries} failed: {last_err}")

    raise RuntimeError(f"Objaverse fetch failed after {retries} attempts: {last_err}")
