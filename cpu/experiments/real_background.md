# Real industrial photographic backgrounds

## Purpose

Compare optimized CPU detection with actual OpenCV on real industrial and warehouse
backgrounds, including frames where the configured detector emits no markers. This
is a dataset preparation record; it establishes no performance result.

## Scope

The 16 real photographs come from Wikimedia Commons. They cover five warehouse/storage
facilities, five manufacturing processes, three machine shops, and three plant/machinery
rooms. There are no generated images, added fiducials, artificial noise, crops, or upscaling.
Photographic content and documented dates/provenance were reviewed before any detector
checks or timing. Sources were chosen by content and source quality, not measured speed.
All selected images remain in the corpus regardless of later results or detections.

Source pages, pinned description-page revision URLs, original image URLs, author/license
attribution, original and prepared hashes, dimensions and selection rationale are in
[`real_background_manifest.json`](real_background_manifest.json). The images themselves
stay under ignored `data/cpu-real-background/` and must not be committed. Each image retains
its own public-domain/CC BY/CC BY-SA status; the repository's Apache-2.0 license does not
license these photographs. Share-alike licenses apply to their respective adapted photos,
not to detector code merely processing them. Attribution and changes must accompany any
later redistribution of photos, under the relevant image license.

## Current state

All 16 originals were acquired and checked against the Commons API's original SHA-1 and
byte size; SHA-256 identities were then pinned. All have an original longest side at least
1901 pixels. The retained test copies apply EXIF orientation, convert to RGB8, and downsize
only when the longest edge exceeds 1920 pixels. The short side uses nearest integer
rounding with ties to even. Resampling is Pillow 10.2.0 LANCZOS; output is RGB8 PNG with
compression level 6 and optimize disabled. One native 1901x1494 image is kept at native
resolution; the other 15 have longest edge 1920. The benchmark performs its ordinary
OpenCV `imread(..., IMREAD_GRAYSCALE)` decode identically for both detector routes,
outside timing. Full-resolution (ArUco3-disabled) measurements use these processed PNG
dimensions, not the original camera resolutions.

Original files are kept in `originals/`. Review sheets and discovery metadata are nested
under `discovery/`, so enumerating top-level PNG/JPEG files selects exactly the 16 inputs.
The contact sheet confirms the intended photographic content, but is not exhaustive marker
annotation. It does not establish that every dictionary/settings combination emits zero
markers. Runtime-zero subsets must be identified from the actual configured OpenCV output,
reported separately, and not used to remove unfavorable inputs from the full corpus.

| ID | Prepared dimensions | Author/credit | Image license | Pinned source page |
| --- | --- | --- | --- | --- |
| `01_warehouse_racks` | 1920x1496 | Axisadman | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=956611489) |
| `02_warehouse_fema` | 1920x1275 | Barry Bahler | [Public domain](https://commons.wikimedia.org/wiki/Template:PD-USGov-FEMA) | [source](https://commons.wikimedia.org/w/index.php?oldid=1091074561) |
| `03_distribution_sainsburys` | 1901x1494 | Nick Saltmarsh | [CC BY 2.0](https://creativecommons.org/licenses/by/2.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1140307618) |
| `04_steel_pipe_storage` | 1440x1920 | Blonder1984 | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=956611726) |
| `05_warehouse_adamhall` | 1920x1395 | Adamhallgmbh | [CC BY-SA 3.0 de](https://creativecommons.org/licenses/by-sa/3.0/de/deed.en) | [source](https://commons.wikimedia.org/w/index.php?oldid=1183026195) |
| `06_robot_bakery` | 1920x1440 | KUKA Roboter GmbH, Bachmann | [Public domain](https://commons.wikimedia.org/wiki/Template:PD-author) | [source](https://commons.wikimedia.org/w/index.php?oldid=1281897711) |
| `07_production_elevator` | 1920x1285 | Tecnowey | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1227617111) |
| `08_can_manufacturing` | 1920x1349 | JopkeB | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1157615658) |
| `09_tannery` | 1920x1440 | Norbzzzz | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1173466477) |
| `10_yarn_production` | 1920x1280 | Odzir | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1228397292) |
| `11_cnc_facility` | 1920x762 | Antoniusaw | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1226970252) |
| `12_noirlab_workshop` | 1920x1280 | NOIRLab/NSF/AURA/T. Slovinský | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1226970343) |
| `13_haas_lathes` | 1920x1280 | Visitor7 | [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1290510742) |
| `14_weir_machine_room` | 1437x1920 | Vincent de Groot - http://www.videgro.net | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1245722753) |
| `15_brewery_machine_room` | 1920x1440 | Rony Vissers | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=959675916) |
| `16_tower_plant_room` | 1920x1440 | Danielrichardbond | [CC BY-SA 4.0](https://creativecommons.org/licenses/by-sa/4.0) | [source](https://commons.wikimedia.org/w/index.php?oldid=1200383136) |

### Reproduction

Use Python with Pillow 10.2.0. It is a preparation-only dependency, not a detector/runtime
library dependency. Decoder/zlib versions are recorded in the manifest; exact pixel and
file hashes detect differences between builds even if the Pillow version matches.

```bash
python3 cpu/experiments/fetch_real_background.py --contact-sheet
python3 cpu/experiments/fetch_real_background.py --verify-only
```

The fetcher downloads sequentially, verifies every original before publishing it, and
recreates prepared inputs with exact hash checks. Existing files are reused only if their
hash matches. It refuses to overwrite mismatching files, and failed preparation leaves no
partial published file. Transient source HTTP 429/503 responses have bounded retries;
permanent source replacement or removal is an explicit reproducibility failure, never a
reason to silently substitute another photo. No detector runs in this script.

For another destination:

```bash
python3 cpu/experiments/fetch_real_background.py --output-dir data/cpu-real-background-copy
```

The original reviewed sheet is `data/cpu-real-background/discovery/contact-sheet.jpg`.
`--contact-sheet` creates `discovery/contact-sheet-reproduced.jpg`; it refuses to replace an
existing sheet. The review file is not used as input to any performance measurement.

### Source and license observations

Most sources are individually attributed CC BY or CC BY-SA photographs. The FEMA logistics
photo is a documented US government employee photograph, public domain in the United
States. The KUKA robot photograph has an author public-domain release with Wikimedia VRT
permission ticket 2010011610015881. The NOIRLab shop photo has an institutional source and
named photographic credit. The manifest records these distinctions instead of inferring an
image license from Wikimedia's software license. The fetch/preparation script and this
record are original repository-authored Apache-2.0 work; no third-party implementation was
copied.

## Goals

- Lock source/transform hashes before performance checks, and retain every preselected input.
- Verify exact detector outputs against actual OpenCV before accepting latency results.
- Report all images and the runtime-zero subset separately when necessary.
- Keep local photos, review images and generated outputs out of the commit.

## Open questions

- Marker absence is not independently annotated; zero output is configuration-specific.
- Still photos do not establish camera-sequence throughput, motion blur robustness or field accuracy.
- This small deliberately diverse collection is not a statistical sample of all industrial scenes.
- Current download URLs can be removed/replaced; pinned hashes fail explicitly in that case.
