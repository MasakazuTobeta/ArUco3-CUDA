#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sample: detect markers in a PGM image with the device-resident detector.

The Python counterpart of examples/detect_image.cpp, with the same options and
the same output, so the two can be compared line by line.

It shows the whole API in the order a caller uses it: size the workspace from
the input, initialize once, upload the image, issue the detection on a stream,
read the results while they are still on the device, and only then bring them
back to the host. The steps that cost a synchronization are marked in the
comments, because those are the ones worth being able to point at.

Usage:
    generate_marker.py --id 42 --output marker.pgm
    detect_image.py --input marker.pgm
"""
import argparse
import sys

import aruco3cuda

import pgm


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers in a binary PGM image."
    )
    parser.add_argument("--input", help="Input image, binary PGM (P5). Required")
    parser.add_argument("--dictionary", default="DICT_ARUCO_MIP_36h12", help="Dictionary name")
    parser.add_argument(
        "--no-aruco3",
        action="store_true",
        help=(
            "Disable the ArUco3 strategy and corner refinement. These two travel "
            "together: the detector rejects any other combination of the pair"
        ),
    )
    parser.add_argument(
        "--default-stream",
        action="store_true",
        help=(
            "Use the default stream instead of an explicit one. CUDA cannot "
            "capture a graph on it, so this is the slower path"
        ),
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Run the detection n times. Useful for showing that the workspace stops allocating",
    )
    arguments = parser.parse_args(argv)

    if not arguments.input:
        print("--input was not specified", file=sys.stderr)
        return 1
    if arguments.repeat < 1:
        print(f"--repeat must be a positive integer: {arguments.repeat}", file=sys.stderr)
        return 1

    try:
        width_px, height_px, pixels = pgm.read_pgm(arguments.input)
    except (OSError, pgm.PgmError) as error:
        print(error, file=sys.stderr)
        return 1

    # Size the workspace from this image rather than leaving the default of
    # 3840x2160. The workspace is allocated once for the worst case these limits
    # imply, so overstating them costs memory on every run.
    config = aruco3cuda.Config(max_width_px=width_px, max_height_px=height_px)
    if arguments.no_aruco3:
        # The detector accepts only ArUco3 with refinement, or neither. With
        # ArUco3 on and refinement off the corners would stay in the coordinates
        # of the downscaled image.
        config.use_aruco3_detection = 0
        config.corner_refine_method = aruco3cuda.CornerRefine.NONE

    try:
        with aruco3cuda.Detector() as detector:
            # Synchronizes the whole device, once, so that the dictionary
            # transfer is visible from every stream afterwards.
            detector.initialize(arguments.dictionary, config)

            with aruco3cuda.DeviceImage.from_host(pixels, width_px, height_px) as image:
                stream = None if arguments.default_stream else aruco3cuda.Stream()
                try:
                    for _ in range(arguments.repeat):
                        # Issues the kernels and returns; nothing has run yet.
                        detector.detect(image, stream)

                        # The results are already addressable on the device here.
                        # A pose estimation stage on the same device would read
                        # them from this object and never touch the host.
                        detector.device_detections()

                        # Waits for the stream. In steady state this is the only
                        # synchronization point, and it exists only because this
                        # sample prints from the host.
                        result = detector.download(stream)
                finally:
                    if stream is not None:
                        stream.close()

                workspace = detector.workspace_statistics()
    except aruco3cuda.Error as error:
        print(error, file=sys.stderr)
        return 1

    info = aruco3cuda.find_dictionary(arguments.dictionary)
    refinement = "kNone" if config.corner_refine_method == 0 else "kSubpix"
    print(f"aruco3cuda   : {aruco3cuda.version()}")
    print(f"input        : {arguments.input} ({width_px}x{height_px})")
    print(
        f"dictionary   : {info.name} ({info.marker_size}x{info.marker_size} bits, "
        f"{info.code_count} ids)"
    )
    print(
        f"configuration: ArUco3 {'on' if config.use_aruco3_detection else 'off'}, "
        f"corner refinement {refinement}, "
        f"{'default stream' if arguments.default_stream else 'explicit stream'}, "
        f"{arguments.repeat} iteration(s)"
    )

    if config.use_aruco3_detection:
        # The single most common reason for finding nothing: the ArUco3 strategy
        # segments a downscaled image, so a marker below this side length in the
        # original image cannot be found at all.
        lower_bound_px = config.min_side_length_canonical_img_px + max(
            width_px, height_px
        ) * config.min_marker_length_ratio_original_img
        print(f"detectable   : markers of side {lower_bound_px:.1f} px or more in this image")

    print(
        f"workspace    : {workspace.allocation_count} allocation(s), "
        f"{workspace.reallocation_count} reallocation(s), peak "
        f"{workspace.peak_used_bytes / (1024.0 * 1024.0):.2f} MiB"
    )
    print(f"detections   : {len(result)} (accepted {result.accepted_total})")
    if result.truncated:
        print(f"  the results were truncated at max_markers_ = {config.max_markers}")
    for index, detection in enumerate(result):
        corners = " ".join(f"({x:.2f}, {y:.2f})" for x, y in detection.corners)
        print(f"  [{index}] id={detection.id} rotation={detection.rotation} corners= {corners}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
