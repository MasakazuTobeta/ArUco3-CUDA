#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Sample: render a marker from a built-in dictionary and write it as a PGM.

The Python counterpart of examples/generate_marker.cpp. It follows the same
rendering rule and takes the same options, so the two can be compared directly;
a test checks that they produce identical bytes.

Provenance:
    The rendering rule is the one OpenCV 4.x uses in
    cv::aruco::Dictionary::generateImageMarker (Apache-2.0): the border cells are
    black, an inner bit of 1 is white and 0 is black, and the cell grid is scaled
    up by nearest neighbour. Only the observable rule is reproduced; no OpenCV
    code is copied, and nothing here derives from the GPLv3 ArUco distribution.
    See docs/ip-and-licensing.md.

Usage:
    generate_marker.py --id 42 --size 200 --margin 40 --output marker.pgm
"""
import argparse
import sys

import aruco3cuda

import pgm

kWhite = 255
kBlack = 0


def render_marker(dictionary_name, marker_id, border_bits, side_px):
    """Render one marker at the requested size, without the quiet zone.

    The cell grid is (marker_size + 2 * border_bits) on a side. Each output pixel
    takes the cell that covers it, which is nearest-neighbour scaling: cell
    boundaries land exactly on pixel boundaries only when side_px is a multiple
    of the cell count, and otherwise some cells come out one pixel wider. That is
    the same behaviour as an INTER_NEAREST resize, and it is why a size that is a
    multiple of the cell count gives the cleanest marker.
    """
    bits = aruco3cuda.marker_bits(dictionary_name, marker_id)
    marker_size = len(bits)
    cells = marker_size + 2 * border_bits
    if side_px < cells:
        raise ValueError(f"--size must be at least the cell count ({cells}): {side_px}")

    pixels = bytearray(side_px * side_px)
    for row_px in range(side_px):
        cell_row = row_px * cells // side_px
        inner_row = cell_row - border_bits
        row_base = row_px * side_px
        for col_px in range(side_px):
            cell_col = col_px * cells // side_px
            inner_col = cell_col - border_bits
            value = kBlack
            if 0 <= inner_row < marker_size and 0 <= inner_col < marker_size:
                value = kWhite if bits[inner_row][inner_col] else kBlack
            pixels[row_base + col_px] = value
    return pixels


def add_quiet_zone(marker, marker_side_px, margin_px):
    """Place the rendered marker at the centre of a white canvas."""
    side_px = marker_side_px + 2 * margin_px
    canvas = bytearray([kWhite]) * (side_px * side_px)
    for row_px in range(marker_side_px):
        source = row_px * marker_side_px
        destination = (row_px + margin_px) * side_px + margin_px
        canvas[destination : destination + marker_side_px] = marker[
            source : source + marker_side_px
        ]
    return side_px, canvas


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Render an ArUco marker from a built-in dictionary as a PGM.",
        epilog=(
            "The quiet zone is not decoration. A marker whose black border touches "
            "the image edge is rejected by min_distance_to_border_px, so a margin "
            "of 0 produces an image that detect_image.py will find nothing in."
        ),
    )
    parser.add_argument("--output", help="Destination PGM file. Required unless listing")
    parser.add_argument("--dictionary", default="DICT_ARUCO_MIP_36h12", help="Dictionary name")
    parser.add_argument("--id", type=int, default=0, help="Marker id. Default 0")
    parser.add_argument(
        "--size", type=int, default=200, help="Side of the marker itself, in pixels"
    )
    parser.add_argument(
        "--margin", type=int, default=40, help="White quiet zone on each side, in pixels"
    )
    parser.add_argument(
        "--border-bits", type=int, default=1, help="Width of the black border, in cells"
    )
    parser.add_argument(
        "--list-dictionaries", action="store_true", help="List the built-in dictionaries and exit"
    )
    arguments = parser.parse_args(argv)

    if arguments.list_dictionaries:
        for info in aruco3cuda.dictionaries():
            print(
                f"{info.name}  {info.marker_size}x{info.marker_size} bits, "
                f"{info.code_count} ids"
            )
        return 0

    if not arguments.output:
        print("--output was not specified", file=sys.stderr)
        return 1
    # argv is external input and is not trusted, so the ranges are checked here
    # rather than left to fail somewhere inside the rendering loop.
    if not 1 <= arguments.size <= pgm.kMaxSidePx:
        print(f"--size must be between 1 and {pgm.kMaxSidePx}: {arguments.size}", file=sys.stderr)
        return 1
    if not 0 <= arguments.margin <= pgm.kMaxSidePx:
        print(
            f"--margin must be between 0 and {pgm.kMaxSidePx}: {arguments.margin}",
            file=sys.stderr,
        )
        return 1
    if not 1 <= arguments.border_bits <= 16:
        print(f"--border-bits must be between 1 and 16: {arguments.border_bits}", file=sys.stderr)
        return 1
    if arguments.size + 2 * arguments.margin > pgm.kMaxSidePx:
        print(f"--size plus twice --margin exceeds {pgm.kMaxSidePx}", file=sys.stderr)
        return 1

    try:
        marker = render_marker(
            arguments.dictionary, arguments.id, arguments.border_bits, arguments.size
        )
    except (aruco3cuda.Error, ValueError) as error:
        print(error, file=sys.stderr)
        return 1

    side_px, canvas = add_quiet_zone(marker, arguments.size, arguments.margin)
    try:
        pgm.write_pgm(arguments.output, side_px, side_px, canvas)
    except (OSError, pgm.PgmError) as error:
        print(error, file=sys.stderr)
        return 1

    print(
        f"wrote {arguments.output}: {side_px}x{side_px}, {arguments.dictionary} "
        f"id={arguments.id}, marker {arguments.size} px, margin {arguments.margin} px"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
