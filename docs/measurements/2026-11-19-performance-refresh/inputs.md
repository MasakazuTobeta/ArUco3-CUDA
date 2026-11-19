# Performance refresh inputs

## Purpose

Reuse the established ArUco and AprilTag inputs and extend the comparison to
changing 1080p and 4K scenes and native high-resolution industrial photographs.

## Scope

The synthetic inputs use DICT_ARUCO_MIP_36h12 or tag36h11. The renderer provides
marker IDs and oriented corners. The industrial photographs have no marker
annotations. Record detector outputs on those photographs without treating
zero returned detections as independently verified marker absence.

## Current state

### Existing inputs

| Local input directory | Preparation |
| --- | --- |
| aruco-steady | Seven synthetic scenes and 16 industrial photographs from the [pinned timing manifest](../../../tools/gpu_quality/steady-inputs.json) |
| aruco-historical | The [28-scene sweep](../../benchmark-report.md#reproducing-the-measurements), selected from the prepared cpu-extended corpus. The shared files follow the seeded [CPU generator procedure](../../cpu-benchmark-report.md#reproduction) |
| corpus-36h11, corpus-720-36h11 | 96 scenes each, rendered at 640×480 and 1280×720 with the [AprilTag procedure](../../../apriltag/evaluation/README.md) |
| april-field, tag-damage, nvidia-fixture | The [pinned outdoor positions, payload damage controls and NVIDIA integration image](../../../apriltag/evaluation/README.md) |
| april-photos | The same 16 prepared industrial PNGs as ArUco, decoded with OpenCV 4.14.0 IMREAD_GRAYSCALE and written as PGM |

The original AprilTag 640×480 and 1280×720 manifest SHA-256 values are
a3b05a54dfbda10f7c2781e6a2971975589e60706f8f929cdc109807a8a4c9e2 and
23b85feec9f84700ca515206cb2c6ded05a870d0866d07bb8a348463c0435c07.

### Higher-resolution synthetic inputs

Use the [CPU quality renderer](../../../tools/cpu_quality/quality_scenes.md)
at 46abcad44c1e48dd54f1e3f80feb4d6fb2150eab with the pinned OpenCV 4.14.0
revision 0654a42e19215ef25b1d367d822f3c630447e7c7. The measured corpus was
prepared with GCC 13 Release in the entry-point environment from the
aruco3-cuda:dgx-spark image with ID
sha256:2840932a0c9225c1ba1de1625e0e87ead55c68a4d64a63451a4e6c20432ad62d.
Generate once and distribute the same files across hosts. Different OpenCV
builds can render different pixels from the same seed.

```bash
input_root=build/performance-refresh-20261119/inputs
renderer=build/quality/cpu_quality/aruco3cuda_cpu_quality_corpus
selector=docs/measurements/2026-11-19-performance-refresh/select_balanced.py
mkdir -p "$input_root"
export OPENCV_FOR_THREADS_NUM=2
"$renderer" "$input_root/aruco-1080" 1920 1080
"$renderer" "$input_root/aruco-2160" 3840 2160
"$renderer" "$input_root/tag-1080" 1920 1080 DICT_APRILTAG_36h11
"$renderer" "$input_root/tag-2160" 3840 2160 DICT_APRILTAG_36h11
for dataset in aruco-1080 aruco-2160 tag-1080 tag-2160; do
  python3 "$selector" "$input_root/$dataset" "$input_root/$dataset-balanced"
done
```

Each full manifest has 96 scenes and 136 rendered markers. The
[selection script](select_balanced.py) retains four scenes from each of eight
families in manifest order. That gives 32 scenes and 48 markers. The choice
was made before timing to fit available disk space while retaining clean
positives, groups, markers around the ArUco3 size threshold, degraded images
and negative controls. The sequence profiles use the two ArUco subsets, the
full 1080p AprilTag set and the 4K AprilTag subset.

| Input directory | Full manifest SHA-256 | Balanced manifest SHA-256 |
| --- | --- | --- |
| aruco-1080 | c1d3ca22322374d31e978625b82335899abc9dbc1703c74d16656421d0d82777 | f2ef49b595b36e2590e3f77b07c93f7516d7d3df573ceca6ca789d799927e457 |
| aruco-2160 | d872790553f5533ddbccc19e85d05c0f25aa0c1a99251abc8efbd71e415a2531 | da215c7f4ac3366c02271b4586aaed2fce2de3381c20927f1066d345d4385622 |
| tag-1080 | 32d877b09392ca51ebe26836c496d1e83a5f7997d8235b8cb60136ac077ce809 | 18e8d98b5e0b7b85d4a67a0f572344f7f297605e6341b839ae73165b47d6dafc |
| tag-2160 | 3799e3a3b79ca6e4f12e00be8f606b29ddb49851199e3b7b293a94ad5f73c18b | 49446e87c2c8b628e93f56cdc36a3f199e2f58d2273392577d4bd4925fd65744 |

Score the ArUco audit with [score_aruco.py](score_aruco.py), Pillow and
the [ground-truth matcher](../../../apriltag/evaluation/README.md#goals).
Keep the audit JSONL in the run's `quality/` directory beside the parent
directory's `commands.jsonl` and `inventory.json`.

```bash
python3 docs/measurements/2026-11-19-performance-refresh/score_aruco.py \
  "$input_root/aruco-2160-balanced/manifest.json" \
  "$run_dir/quality/aruco-2160-balanced-1.jsonl" \
  --scorer build/tag-score --aruco3 1
```

The scorer checks the encoded file hashes and decoded Gray8 pixel hashes,
verifies replayed detections and pairs each output with the rendered truth.
It uses 0.001 pixels for the direct OpenCV corner comparison and for the
common-marker change in corner error against ground truth.

### Native industrial photographs

Fetch originals with the [industrial photograph procedure](../../../cpu/experiments/real_background.md#reproduction)
and verify their bytes against the [pinned source manifest](../../../cpu/experiments/real_background_manifest.json).
The native set includes every original whose oriented longest edge is at
least 3840 pixels. Before measurements, select #04 for the portrait spot
test and #15 for the 8K spot test. The other originals with edges of
3872–5312 pixels remain available in the 4–5K set.

| Input | Width × height | Attribution and license |
| --- | ---: | --- |
| 04_steel_pipe_storage | 3456×4608 | Blonder1984, CC BY-SA 4.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=956611726) |
| 07_production_elevator | 3872×2592 | Tecnowey, CC BY-SA 3.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=1227617111) |
| 10_yarn_production | 4368×2912 | Odzir, CC BY-SA 4.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=1228397292) |
| 11_cnc_facility | 4032×1600 | Antoniusaw, CC BY-SA 4.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=1226970252) |
| 12_noirlab_workshop | 5312×3541 | NOIRLab/NSF/AURA/T. Slovinský, CC BY 4.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=1226970343) |
| 13_haas_lathes | 4608×3072 | Visitor7, CC BY-SA 3.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=1290510742) |
| 15_brewery_machine_room | 8256×6192 | Rony Vissers, CC BY 4.0. [Source](https://commons.wikimedia.org/w/index.php?oldid=959675916) |

Decode directly from the original JPEG using OpenCV 4.14.0 IMREAD_GRAYSCALE.
Retain the oriented native dimensions. Write both the PNG for ArUco and the
PGM for AprilTag from the same Gray8 matrix. The same conversion can be
reproduced in the pinned OpenCV C++ environment by calling
`cv::imread(jpeg, cv::IMREAD_GRAYSCALE)` and then
`cv::imwrite(destination, gray)` with default options for each format.
The prepared 1920-pixel industrial corpus above is a separate input set.

| Input | Gray8 PGM SHA-256 | Gray8 PNG SHA-256 |
| --- | --- | --- |
| 04 | 528ce4bea49e9da8c7e413304ab7daaef7a2afd6d6adf37483c94f2c0944f743 | bf0215b34651ec51acaee6ef78cf6cd750504885c9fbc3d3f82bbec77620f60a |
| 07 | 7646ff3ae4936b344415604cbc6bb33f7d8dbc3b3edf389a60c065ecbfeba8e2 | afc016a73a16dff5ad1b0645ccbadd5d4e9a59ac4310fe4bdb18bae7b28bc8d0 |
| 10 | 722a0bc1965f4c28ad9e112c013f3b4cec77c7c149f51065ee1dfe9f36b5b2ea | 5d16130a63ff137b8e433d4749446cbe1ab6fe804e462a1333d5c0a1787bb8b5 |
| 11 | 797fdc17cfd6a0add7470f4367a0a5639337ad8af770b103d7af7b8d5546c3ba | 29a83a7003b1b34ae60917c9fa7b48e15af1c8da42fee19295d6d1891d9696f3 |
| 12 | 28f429d55e21dd6e0281c230bfad90f5904a311a8a85bb4af04afd3baa6971c6 | 2df5b30b908f5b022de6b1cae4ed724436638fbbc6356a0be2c8c30319a2e5a6 |
| 13 | 8d25dfd17def9b66d7652ac9c1f8f9ae3f39f9b93e8aeab635ce3e695b40122e | 3d152cf4751d4ce82f358c009ec3b9c10895d0adedba350ef27aa5194227c679 |
| 15 | 6061dcd0a881199b637bd7d317c507cf5a5177dceb225c1ff2287163ab8b1bdf | 66894353dc02f1a4404bc47797f027539fc26e2cd1cddff7c2f2df6594293a09 |

Place #15 in industrial-8k and the others in industrial-4k5k. Each directory's
`input-manifest.json` lists PNGs in an `inputs` array with `name`, `sha256` and
`family` set to `industrial photograph`. For AprilTag, create one
`manifest-STEM.json` per PGM with a `scenes` array containing one object with
`name`, `path`, `sha256`, `width_px` and `height_px`. Set `dictionary` to
`DICT_APRILTAG_36h11` when passing the manifest directly to `run_sequence.py`.
The refresh runner supplies it when staging manifests. Each photo is measured
as a single fixed-size scene.

Image files, created input manifests and per-run output remain in [ignored
measurement storage](../README.md). The native photographs retain their
individual source licenses.

## Goals

Compare implementations on identical prepared pixels, cover supported
resolutions and retain the preselected negative and positive scenes.

## Open questions

Which changes in recall and latency at these resolutions also occur on
annotated high-resolution camera sequences?
