# SPDX-License-Identifier: Apache-2.0
"""Select the predeclared 32 scenes in their original manifest order."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil


INDICES = {
    "clean_pose": (0, 1, 4, 7),
    "projective": (0, 3, 6, 11),
    "blur_noise": (0, 5, 10, 15),
    "lighting": (0, 5, 6, 11),
    "size_bound": (0, 4, 7, 11),
    "border_occlusion": (0, 5, 6, 11),
    "multi_dense": (0, 1, 2, 11),
    "negative": (0, 3, 4, 5),
}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()

    manifest_bytes = (args.source / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    chosen = {f"quality_{family}_{i}" for family, indices in INDICES.items() for i in indices}
    scenes = [scene for scene in manifest["scenes"] if scene["name"] in chosen]
    if len(scenes) != len(chosen) or {scene["name"] for scene in scenes} != chosen:
        raise ValueError("source manifest does not contain the expected 32 scenes")

    args.destination.mkdir()
    for scene in scenes:
        name = Path(scene["path"])
        if name.name != scene["path"]:
            raise ValueError(f"expected a filename in the source manifest: {name}")
        target = args.destination / name
        shutil.copyfile(args.source / name, target)
        if hashlib.sha256(target.read_bytes()).hexdigest() != scene["sha256"]:
            raise ValueError(f"wrong input hash for {target}")
    manifest["scenes"] = scenes
    manifest["subset_source_manifest_sha256"] = hashlib.sha256(manifest_bytes).hexdigest()
    (args.destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n",
                                                     encoding="utf-8")


if __name__ == "__main__":
    main()
