import os, random
from math import radians
from pathlib import Path
import objaverse

import signal
from contextlib import contextmanager


class TimeoutError(RuntimeError):
    pass

@contextmanager
def time_limit(seconds: int):
    """
    Raise TimeoutError if the `with` block runs longer than `seconds`.
    Works on Unix (Linux/macOS) and only in the main thread.
    """
    if seconds is None or seconds <= 0:
        yield
        return

    def _handler(signum, frame):
        raise TimeoutError(f"Timed out after {seconds}s")

    old_handler = signal.signal(signal.SIGALRM, _handler)
    try:
        signal.alarm(int(seconds))
        yield
    finally:
        signal.alarm(0)  # cancel alarm
        signal.signal(signal.SIGALRM, old_handler)


def _pick_mesh_file(path_like):
    import zipfile
    exts = {".glb",".gltf",".obj",".fbx"}
    p = Path(path_like)
    if p.is_file() and p.suffix.lower() in exts:
        return str(p)
    if p.is_file() and p.suffix.lower()==".zip":
        with zipfile.ZipFile(p,"r") as zf:
            names = zf.namelist()
            for ext in [".glb",".gltf",".obj",".fbx"]:
                for n in names:
                    if n.lower().endswith(ext):
                        out = p.parent / Path(n).name
                        if not out.exists():
                            zf.extract(n, p.parent)
                            (p.parent / n).rename(out)
                        return str(out)
        raise RuntimeError(f"No supported mesh in zip: {p}")
    if p.is_dir():
        for ext in [".glb",".gltf",".obj",".fbx"]:
            for cand in p.glob(f"*{ext}"): return str(cand)
        for ext in [".glb",".gltf",".obj",".fbx"]:
            for cand in p.rglob(f"*{ext}"): return str(cand)
    raise RuntimeError(f"No supported mesh at/under: {path_like}")
    
def get_random_objaverse(td, timeout_sec=60, retries=3):
    """
    Simple timeout wrapper around objaverse calls.
    - Uses signal.alarm (Unix only, main thread).
    - Retries with a fresh UID up to `retries` times.
    Returns: (uid, mesh_path)
    """
    os.makedirs(td, exist_ok=True)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            # 1) Load UIDs with a timeout
            with time_limit(timeout_sec):
                uids = list(objaverse.load_uids())
            if not uids:
                raise RuntimeError("Objaverse returned no UIDs")

            uid = random.choice(uids)

            # 2) Fetch the object with a timeout
            try:
                with time_limit(timeout_sec):
                    assets = objaverse.load_objects(uids=[uid], download_dir=td)
            except TypeError:
                # Older objaverse versions without download_dir kwarg
                cwd = os.getcwd()
                os.makedirs(td, exist_ok=True)
                try:
                    os.chdir(td)
                    with time_limit(timeout_sec):
                        assets = objaverse.load_objects(uids=[uid])
                finally:
                    os.chdir(cwd)

            if uid not in assets:
                raise RuntimeError(f"No asset for uid {uid}")

            entry = assets[uid]
            if isinstance(entry, str):
                raw_path = entry
            elif isinstance(entry, dict):
                raw_path = entry.get("file_path") or (entry.get("paths")[0] if entry.get("paths") else None)
                if raw_path is None:
                    raw_path = next(
                        (v for v in entry.values() if isinstance(v, str) and os.path.exists(v)),
                        None
                    )
            else:
                raw_path = None

            if not raw_path:
                raise RuntimeError(f"Unrecognized objaverse return type for {uid}")

            mesh_path = _pick_mesh_file(raw_path)  # your existing helper
            if not mesh_path or not os.path.exists(mesh_path):
                raise RuntimeError(f"No mesh file found under {raw_path}")

            return uid, mesh_path

        except TimeoutError as te:
            last_err = str(te)
            print(f"[Objaverse] attempt {attempt}/{retries} timed out; retrying…")
            continue
        except Exception as e:
            last_err = f"{type(e).__name__}: {e}"
            print(f"[Objaverse] attempt {attempt}/{retries} failed: {last_err}; retrying…")
            continue

    raise RuntimeError(f"Objaverse fetch failed after {retries} attempts (last error: {last_err})")
