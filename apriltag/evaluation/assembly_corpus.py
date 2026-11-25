# SPDX-License-Identifier: Apache-2.0
"""Render a held-out sensor-effects corpus on eight NASA assembly and test photographs."""
import argparse
import hashlib
import json
from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook
from industrial_corpus import render_clip


SEED = 170503817
# NASA Image and Video Library asset ID, original/large rendition, and camera image digest.
PHOTOS = (
    ('MSFC-1701231', 'large.jpg', '7c2bd8f3b3abfb1658b9503d7ddbd982c05982f3a1191f112e86c82b9deb2e6c'),
    ('GRC-1970-C-03022', 'orig.jpg', '112031c7f8a3259343c56f0c6d3c789498ba30ca7cce4607ad16fdd2143991ef'),
    ('LRC-2013-00757', 'large.jpg', '5cfdaa3d5af1d6f2e1913f25163143aa348007584363e7cde4ef76ae87c27438'),
    ('PIA11188', 'orig.jpg', '0eb8d44213caaa553f853433857d70035cf48f99b5c5cf700fa3490457582ba1'),
    ('GRC-1987-C-00568', 'orig.JPG', 'bc8c7494ceed28c1f983110edcaa5c8884e27da5942bd0f43d806655274df4c9'),
    ('AAL-4479', 'orig.jpg', '0becace683672929e2c829ff6a97b13b081a4c46e727c3ba4f4717ea3690f0dd'),
    ('PIA12020', 'orig.jpg', '3866a53d5b61618cdccc3ef78b837aed1f89162a2feae707076e4aba8c254bfb'),
    ('KSC-20201021-PH-JBS02_0043', 'orig.jpg', 'a65a7026fa9899dd871e65ea6e3b395a575b53b206af34671aa55eb0fe463429'),
)

VALIDATION_PHOTOS = (
    ('AFRC2019-0134-03', 'large.jpg', '90776225fb33c0dc2e563a10db2776c2a1fadb67c5b468d405544ff709ae2c92'),
    ('GRC-2011-C-00334', 'orig.jpg', '930cfc4680d03e16b9e2b3e7a55f8d124f03c044b8b4fe9992cac8939fa4f923'),
    ('jsc2022e043550', 'large.jpg', '6b82361ca36377b2c1d4d09a6cc45911b6884e45b02a68cbc4f77a9ac79c5955'),
    ('PIA12463', 'orig.jpg', '19bece9c379a7b2294300a729f70d7385e81ade1d5d1b85dcff4429533ff3213'),
    ('GRC-1949-C-24341', 'orig.jpg', '5d0af422ec46baa81c69fc0f474dba06515beee5601609992a89b2f77883e1fc'),
    ('A-42536-003', 'large.jpg', 'b84d5a1726d0208ad985845202c6d5263990a78fac454ebb5cc4857f55da29e2'),
    ('PIA09205', 'orig.jpg', '14d2364cb1ac4d12e00f9f94f83abd2e082f382d6913e69077c31b92059258cf'),
    ('KSC-20190320-PH_KLS01_0306', 'large.jpg', '62ee0f8a555222eeb379d6485879dd1b4ac12941cc43cdb25b0d43edc136965f'),
)


def source_image(cache, identifier, rendition, digest, offline):
    path = cache / f'{identifier}.jpg'
    if path.exists() or offline:
        data = path.read_bytes()
    else:
        url = f'https://images-assets.nasa.gov/image/{identifier}/{identifier}~{rendition}'
        with urlopen(url, timeout=60) as response:
            data = response.read()
    if hashlib.sha256(data).hexdigest() != digest:
        raise ValueError(f'NASA source image digest mismatch for {identifier}')
    image = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f'unreadable source image for {identifier}')
    if not path.exists():
        path.write_bytes(data)
    return image


def generate(cache, output, height, offline, validation=False):
    cache.mkdir(parents=True, exist_ok=True)
    output.mkdir(parents=True, exist_ok=True)
    if any(output.iterdir()):
        raise ValueError('output directory must be empty')
    grids = grids_from_codebook(CODEBOOK)
    scenes = []
    photos = VALIDATION_PHOTOS if validation else PHOTOS
    seed = 120736991 if validation else SEED
    for index, (identifier, rendition, digest) in enumerate(photos):
        image = source_image(cache, identifier, rendition, digest, offline)
        # Even renderer indices cover every marker size and all four sensor profiles.
        scenes.extend(render_clip(image, {'name': identifier}, 2 * index, height * 16 // 9,
                                  height, grids, output, seed, True))
    manifest = {'dictionary': 'tag36h11', 'partition': 'assembly-validation' if validation else 'assembly', 'seed': seed,
                'sensor_effects': True, 'sources': {image: digest for image, _, digest in photos},
                'scenes': scenes}
    data = (json.dumps(manifest, indent=2) + '\n').encode()
    (output / 'manifest.json').write_bytes(data)
    print(hashlib.sha256(data).hexdigest(), output / 'manifest.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('cache', type=Path, help='directory for the eight NASA images')
    parser.add_argument('output', type=Path)
    parser.add_argument('--height', type=int, choices=(720, 1080, 2160), required=True)
    parser.add_argument('--offline', action='store_true', help='require all source images in cache')
    parser.add_argument('--validation', action='store_true', help='use the second eight photographs and an independent placement seed')
    args = parser.parse_args()
    cv2.setNumThreads(1)
    generate(args.cache, args.output, args.height, args.offline, args.validation)
