# SPDX-License-Identifier: Apache-2.0
"""Minimal binary PGM (P5) reader and writer for the Python samples.

The same reasoning as examples/pgm.hpp: PGM stores exactly what the detector
takes, 8-bit grayscale, so no conversion step sits between the file on disk and
the API being shown, and the samples stay free of an image library.

A PGM header is untrusted input. The magic number, the dimensions, the maximum
value, and the length of the payload are all checked, and ASCII PGM (P2) and
16-bit PGM are reported by name rather than silently misread.
"""

# Upper bound on the side length accepted, in pixels. Matches kMaxSidePx in
# examples/pgm.hpp so that both sets of samples reject the same inputs.
kMaxSidePx = 20000

kSpace = b" \t\r\n\v\f"


class PgmError(ValueError):
    """Raised when a file is not a PGM this reader supports."""


def _next_token(data, position):
    """Read one whitespace-separated token, skipping comments.

    Returns (token, position_after). Comments run from '#' to end of line and
    may appear between any two tokens. The whitespace byte that ends the token
    is consumed, which is what the format requires: exactly one whitespace byte
    separates the maximum value from the start of the payload.
    """
    while position < len(data):
        if data[position] == 0x23:  # '#'
            while position < len(data) and data[position] != 0x0A:
                position += 1
        elif data[position] in kSpace:
            position += 1
        else:
            break
    start = position
    while position < len(data) and data[position] not in kSpace:
        position += 1
    token = data[start:position]
    if position < len(data):
        position += 1  # consume the single separator
    return token, position


def _parse_non_negative_int(token):
    """Parse a token as a run of decimal digits, or return None.

    int() is not used directly because it accepts a sign, underscores, and
    surrounding whitespace, none of which belong in a PGM header.
    """
    if not token or len(token) > 10 or not token.isdigit():
        return None
    return int(token)


def read_pgm(path):
    """Read a binary PGM file.

    Returns (width_px, height_px, pixels) where pixels is a bytes object of
    width * height entries, rows contiguous. Raises PgmError with the reason,
    or OSError if the file cannot be opened.
    """
    with open(path, "rb") as handle:
        data = handle.read()
    if not data:
        raise PgmError(f"the file is empty: {path}")

    token, position = _next_token(data, 0)
    if token != b"P5":
        if token == b"P2":
            raise PgmError(
                f"ASCII PGM (P2) is not supported. Save it as binary PGM (P5): {path}"
            )
        raise PgmError(f'not a PGM: the magic number is "{token.decode("ascii", "replace")}"')

    values = []
    for _ in range(3):
        token, position = _next_token(data, position)
        value = _parse_non_negative_int(token)
        if value is None:
            raise PgmError(f"the header is malformed: {path}")
        values.append(value)
    width_px, height_px, max_value = values

    if not 1 <= width_px <= kMaxSidePx or not 1 <= height_px <= kMaxSidePx:
        raise PgmError(
            f"the dimensions are out of range: width={width_px} height={height_px}"
        )
    if max_value != 255:
        raise PgmError(f"only a maximum value of 255 is supported: max_value={max_value}")

    expected = width_px * height_px
    pixels = data[position : position + expected]
    if len(pixels) != expected:
        raise PgmError(
            f"the payload is short: expected {expected} bytes but read {len(pixels)}"
        )
    return width_px, height_px, pixels


def write_pgm(path, width_px, height_px, pixels):
    """Write a binary PGM file, overwriting any existing one."""
    if not 1 <= width_px <= kMaxSidePx or not 1 <= height_px <= kMaxSidePx:
        raise PgmError(
            f"the dimensions are out of range: width={width_px} height={height_px}"
        )
    expected = width_px * height_px
    if len(pixels) != expected:
        raise PgmError(
            f"the pixel count does not match the dimensions: expected {expected} "
            f"but held {len(pixels)}"
        )
    with open(path, "wb") as handle:
        handle.write(b"P5\n%d %d\n255\n" % (width_px, height_px))
        handle.write(bytes(pixels))
