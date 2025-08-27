import os, random
from math import radians
from pathlib import Path
import objaverse


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

def get_random_objaverse(td):
    uids = list(objaverse.load_uids())
    if not uids: raise RuntimeError("Objaverse returned no UIDs")
    uid = random.choice(uids)
    try:
        assets = objaverse.load_objects(uids=[uid], download_dir=td)
    except TypeError:
        os.makedirs(td, exist_ok=True)
        cwd = os.getcwd()
        try:
            os.chdir(td); assets = objaverse.load_objects(uids=[uid])
        finally:
            os.chdir(cwd)
    if uid not in assets: raise RuntimeError(f"No asset for uid {uid}")
    entry = assets[uid]
    if isinstance(entry, str):
        raw_path = entry
    elif isinstance(entry, dict):
        raw_path = entry.get("file_path") or (entry.get("paths")[0] if entry.get("paths") else None)
        if raw_path is None:
            for v in entry.values():
                if isinstance(v,str) and os.path.exists(v):
                    raw_path = v; break
    else:
        raw_path = None
    if not raw_path: raise RuntimeError(f"Unrecognized objaverse return type for {uid}")
    return uid, _pick_mesh_file(raw_path)
