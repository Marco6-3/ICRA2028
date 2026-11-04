"""Hash the V2.2 archive and its two owning research documents; never alter history."""
from pathlib import Path
import argparse
import hashlib
import json

ROOT = Path(__file__).resolve().parent
MANIFEST = ROOT / "archive_manifest.json"


def fingerprint(path):
    return {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--verify", action="store_true")
    args = parser.parse_args()
    if not args.verify:
        files = {p.relative_to(ROOT).as_posix(): fingerprint(p) for p in sorted(ROOT.rglob("*"))
                 if p.is_file() and p != MANIFEST and "__pycache__" not in p.parts}
        for name in ["STAGE4_V2_2_CMQ_PROPRIO_RESIDUAL_RL_20261103.md", "ICRA2028_MINIMAL_PAPER_FREEZE.md"]:
            rel = "../../research/" + name
            files[rel] = fingerprint(ROOT / rel)
        MANIFEST.write_text(json.dumps({"archive": ROOT.name, "date": "2026-11-04", "file_count": len(files),
            "scope": "Current source, data, weights, outputs, figures, docs and retained snapshots; manifest itself and Python caches excluded.",
            "external_dependencies": "Existing E1S pure_loading_01/pilot_tangent_02, V2.1/Stage3S weights, Isaac Lab/Sim and official assets remain required.",
            "files": files}, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    stored = json.loads(MANIFEST.read_text(encoding="utf-8"))
    errors = [name for name, sig in stored["files"].items()
              if not (ROOT / name).is_file() or fingerprint(ROOT / name) != sig]
    if errors:
        raise SystemExit("SHA256/size mismatch: " + ", ".join(errors))
    assert stored["file_count"] == len(stored["files"])
    print(f'Verified {stored["file_count"]} file SHA256 and size entries.')


if __name__ == "__main__":
    main()
