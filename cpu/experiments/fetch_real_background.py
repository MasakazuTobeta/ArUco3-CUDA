#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Fetch and verify pinned, local-only real photographic CPU background inputs.

Requires Pillow 10.2.0 for byte-identical preparation. No detector is executed.
Example: python3 cpu/experiments/fetch_real_background.py --contact-sheet
"""
import argparse
import hashlib
import io
import os
import json
import pathlib
import re
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request

from PIL import Image, ImageDraw, ImageFont, ImageOps

USER_AGENT = "ArUco3-CUDA/1.0 (https://github.com/tobeta/ArUco3-CUDA; reproducible research)"
MAX_DOWNLOAD_BYTES = 80 * 1024 * 1024


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify_file(path, expected):
    if path.is_symlink() or not path.is_file():
        raise RuntimeError(f"Expected a regular file: {path}")
    data = path.read_bytes()
    if digest(data) != expected:
        raise RuntimeError(f"SHA-256 mismatch; existing file was not changed: {path}")
    return data


def publish(path, data):
    """Publish a complete file without replacing any existing destination."""
    if path.exists() or path.is_symlink():
        raise RuntimeError(f"Refusing to overwrite: {path}")
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".real-background-", delete=False) as out:
        temporary = pathlib.Path(out.name)
        try:
            out.write(data)
            out.close()
            # A hard link publishes atomically and fails if another producer won.
            os.link(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def download(url):
    if urllib.parse.urlsplit(url).hostname != "upload.wikimedia.org":
        raise RuntimeError("Original URL must use the pinned Wikimedia upload host")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    for attempt in range(4):
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                data = response.read(MAX_DOWNLOAD_BYTES + 1)
            if len(data) > MAX_DOWNLOAD_BYTES:
                raise RuntimeError("Original exceeds download-size limit")
            return data
        except urllib.error.HTTPError as error:
            if error.code not in (429, 503) or attempt == 3:
                raise
            retry = error.headers.get("Retry-After", "")
            delay = min(60, int(retry)) if retry.isdecimal() else 10 * (attempt + 1)
            time.sleep(delay)
    raise RuntimeError("Download attempts exhausted")


def prepare(data, item, transformation):
    with Image.open(io.BytesIO(data)) as source:
        if source.size != (item["original_width"], item["original_height"]):
            raise RuntimeError(f"Original dimensions differ: {item['id']}")
        image = ImageOps.exif_transpose(source).convert("RGB")
        if image.size != (item["oriented_width"], item["oriented_height"]):
            raise RuntimeError(f"Oriented dimensions differ: {item['id']}")
        if max(image.size) > transformation["max_long_edge"]:
            # Target dimensions are pinned and were computed with aspect-preserving rounding.
            image = image.resize((item["width"], item["height"]), Image.Resampling.LANCZOS)
        if image.size != (item["width"], item["height"]):
            raise RuntimeError(f"Prepared dimensions differ: {item['id']}")
        if digest(image.tobytes()) != item["prepared_pixel_rgb_sha256"]:
            raise RuntimeError(f"Prepared pixels differ; check Pillow/decoder build: {item['id']}")
        output = io.BytesIO()
        image.save(output, format="PNG", optimize=False, compress_level=6)
        result = output.getvalue()
    if digest(result) != item["prepared_sha256"]:
        raise RuntimeError(f"PNG encoding differs; check Pillow/zlib build: {item['id']}")
    return result


def contact_sheet(destination, files):
    sheet = Image.new("RGB", (1600, ((len(files) + 3) // 4) * 330), "#eeeeee")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=16)
    for index, item in enumerate(files):
        with Image.open(destination / item["name"]) as image:
            image.thumbnail((392, 290))
            x = (index % 4) * 400 + (400 - image.width) // 2
            y = (index // 4) * 330
            sheet.paste(image, (x, y))
        draw.text(((index % 4) * 400 + 5, (index // 4) * 330 + 296),
                  item["id"], font=font, fill="black")
    review = destination / "discovery"
    if review.is_symlink():
        raise RuntimeError("Review destination must not be a symlink")
    review.mkdir(exist_ok=True)
    path = review / "contact-sheet-reproduced.jpg"
    if path.exists():
        raise RuntimeError(f"Review sheet already exists; choose another output directory: {path}")
    sheet.save(path, quality=94)
    print(f"Review only, never a timing input: {path}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=pathlib.Path,
                        default=pathlib.Path(__file__).with_name("real_background_manifest.json"))
    parser.add_argument("--output-dir", type=pathlib.Path, default=pathlib.Path("data/cpu-real-background"))
    parser.add_argument("--verify-only", action="store_true", help="verify all existing images without network or writes")
    parser.add_argument("--contact-sheet", action="store_true", help="create a separate review-only contact sheet")
    args = parser.parse_args()
    if args.verify_only and args.contact_sheet:
        parser.error("--verify-only cannot be combined with --contact-sheet")
    manifest_bytes = args.manifest.read_bytes()
    manifest = json.loads(manifest_bytes)
    files = manifest["files"]
    destination = args.output_dir
    if destination.is_symlink() or (destination.exists() and not destination.is_dir()):
        raise RuntimeError("Output directory must be a real directory")
    if not args.verify_only:
        if Image.__version__ != manifest["transformation"]["version"]:
            raise RuntimeError("Reproduction requires the recorded Pillow version; existing files may use --verify-only")
        destination.mkdir(parents=True, exist_ok=True)
    originals = destination / "originals"
    if originals.is_symlink():
        raise RuntimeError("Originals directory must not be a symlink")
    if not args.verify_only:
        originals.mkdir(exist_ok=True)
    names = set()
    for item in files:
        if not re.fullmatch(r"[0-9a-z_]+\.png", item["name"]) or item["name"] in names:
            raise RuntimeError("Unsafe or duplicate prepared filename")
        names.add(item["name"])
        if not re.fullmatch(r"originals/[0-9a-z_]+\.(jpg|png)", item["original_filename"]):
            raise RuntimeError("Unsafe original filename")
        original = destination / item["original_filename"]
        if original.exists() or original.is_symlink() or args.verify_only:
            data = verify_file(original, item["original_sha256"])
        else:
            data = download(item["original_url"])
            if len(data) != item["original_size_bytes"] or digest(data) != item["original_sha256"]:
                raise RuntimeError(f"Downloaded bytes differ from the locked manifest: {item['id']}")
            publish(original, data)
            time.sleep(2)  # Sequential requests avoid overwhelming the source host.
        target = destination / item["name"]
        if target.exists() or target.is_symlink() or args.verify_only:
            verify_file(target, item["prepared_sha256"])
            with Image.open(target) as image:
                if image.size != (item["width"], item["height"]) or digest(image.convert("RGB").tobytes()) != item["prepared_pixel_rgb_sha256"]:
                    raise RuntimeError(f"Prepared pixel/dimension mismatch: {target}")
        else:
            publish(target, prepare(data, item, manifest["transformation"]))
        print(f"Verified {item['id']}: {item['width']}x{item['height']} {item['prepared_sha256']}")
    print(f"Manifest SHA-256: {digest(manifest_bytes)}; all {len(files)} inputs verified")
    if args.contact_sheet:
        contact_sheet(destination, files)


if __name__ == "__main__":
    main()
