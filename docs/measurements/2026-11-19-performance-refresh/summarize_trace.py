# SPDX-License-Identifier: Apache-2.0
"""Summarize the last detection in a single-input Nsight Systems CUDA trace."""

import argparse
import json
import sqlite3
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path, help="exported Nsight Systems SQLite database")
    parser.add_argument("--upload-bytes", type=int, required=True)
    parser.add_argument("--expected-uploads", type=int, required=True)
    args = parser.parse_args()
    with sqlite3.connect(args.trace.resolve().as_uri() + "?mode=ro", uri=True) as db:
        transfers = db.execute("""SELECT end, deviceId, contextId
                                  FROM CUPTI_ACTIVITY_KIND_MEMCPY
                                  WHERE copyKind = 1 AND bytes = ?
                                  ORDER BY end""", (args.upload_bytes,)).fetchall()
        if len(transfers) != args.expected_uploads:
            raise ValueError(f"expected {args.expected_uploads} input uploads, found {len(transfers)}")
        count, duration = db.execute("""SELECT count(*), sum(end - start)
                                       FROM CUPTI_ACTIVITY_KIND_KERNEL
                                       WHERE start >= ? AND deviceId = ? AND contextId = ?""",
                                     transfers[-1]).fetchone()
        if count < 1 or duration <= 0:
            raise ValueError("no positive-duration kernels following the final input upload")
    print(json.dumps({"input_uploads": len(transfers), "kernel_count": count,
                      "summed_kernel_time_ms": duration / 1e6}, indent=2))


if __name__ == "__main__":
    main()
