"""Re-copy the shared measurements from a tagged Stemslayer checkout.

Editing the vendored files by hand is the failure this exists to prevent. Two
definitions of "absent" would let a checkpoint pass evaluation here and be
rejected by admission there, which makes the admission gate meaningless.

    python Tools/refresh_vendored_metrics.py <path-to-separador-pistas> v1.3.0
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path

VENDOR = Path(__file__).resolve().parent.parent / "RoleSpecialist" / "vendor"
FILES = {
    "role_metrics.py": "SeparationWorker/engine/role_metrics.py",
    "thresholds.json": "Compliance/evidence/metal-guitar/thresholds.json",
}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    repository, tag = Path(argv[0]), argv[1]

    provenance = json.loads((VENDOR / "PROVENANCE.json").read_text(encoding="utf-8"))
    entries = {}
    for name, origin in FILES.items():
        # Read the tagged bytes, never the working tree: a dirty checkout would
        # vendor something no release ever contained.
        content = subprocess.run(
            ["git", "show", f"{tag}:{origin}"],
            cwd=repository,
            check=True,
            capture_output=True,
        ).stdout
        (VENDOR / name).write_bytes(content)
        entries[name] = {"origin_path": origin, "sha256": hashlib.sha256(content).hexdigest()}
        print(f"{name}: {entries[name]['sha256']}")

    provenance["source_tag"] = tag
    provenance["files"] = entries
    (VENDOR / "PROVENANCE.json").write_text(json.dumps(provenance, indent=2) + "\n", encoding="utf-8")
    print(f"vendored from {tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
