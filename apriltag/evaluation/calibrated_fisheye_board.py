# SPDX-License-Identifier: Apache-2.0
"""Render a planar tag36h11 board in the native pixels of a calibrated fisheye camera.

python3 -m pip install numpy opencv-python-headless
python3 apriltag/evaluation/calibrated_fisheye_board.py /tmp/aist-board --split held

Projected physical corners and edge midpoints use OpenCV's forward fisheye model.
Image brightness uses independent inverse projection and subpixel ray-plane intersection.
"""

from dataclasses import dataclass
from pathlib import Path
import argparse
import hashlib
import json

import cv2
import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook


@dataclass(frozen=True)
class Camera:
    name: str
    source: str
    width: int
    height: int
    matrix: np.ndarray
    distortion: np.ndarray
    wide_rays: bool = False


# Published LUMIX DMC-GX8 8 mm calibration verified at stella-cv/stella_vslam
# commit e445b5452535e781fdf6777ec33a06f5f8f5e416, example/aist/fisheye.yaml.
AIST = Camera(
    "AIST LUMIX DMC-GX8 8 mm",
    "https://github.com/stella-cv/stella_vslam/blob/main/example/aist/fisheye.yaml",
    960,
    540,
    np.array(
        [
            [441.730011169, 0, 480.35667528],
            [0, 442.520822476, 275.490228646],
            [0, 0, 1.0],
        ]
    ),
    np.array(
        [
            -0.03068701296607466433,
            -0.003343454364086094217,
            -0.002881735840896060968,
            -0.0005917420310474077278,
        ]
    ),
)
# TUM VI's official pinhole-equi-512.zip, cam0 in camchain-imucam-imucalib.yaml.
TUM_VI = Camera(
    "TUM VI cam0 512 KB4",
    "https://cvg.cit.tum.de/data/datasets/visual-inertial-dataset",
    512,
    512,
    np.array(
        [
            [190.97847715128717, 0, 254.93170605935475],
            [0, 190.9733070521226, 256.8974428996504],
            [0, 0, 1.0],
        ]
    ),
    np.array(
        [
            0.0034823894022493434,
            0.0007150348452162257,
            -0.0020532361418706202,
            0.00020293673591811182,
        ]
    ),
    wide_rays=True,
)
SS = 4


def project(camera, xyz):
    return cv2.fisheye.projectPoints(
        np.ascontiguousarray(xyz, np.float64).reshape(-1, 1, 3),
        np.zeros(3),
        np.zeros(3),
        camera.matrix,
        camera.distortion,
    )[0].reshape(xyz.shape[:-1] + (2,))


def native_camera_rays(camera):
    # Pixel coordinates in the calibration are pixel centers; area-sample each native pixel.
    x = (np.arange(camera.width * SS) + 0.5) / SS - 0.5
    y = (np.arange(camera.height * SS) + 0.5) / SS - 0.5
    sensor = np.stack(np.meshgrid(x, y), axis=-1).reshape(-1, 1, 2)
    if camera.wide_rays:
        # The perspective plane used by undistortPoints cannot represent the
        # rays beyond 90 degrees in this sensor's corners. Invert the calibrated
        # angle polynomial directly, then intersect the 3-D unit ray with the board.
        native = sensor.reshape(camera.height * SS, camera.width * SS, 2)
        matrix = camera.matrix
        xy = (native - np.array([matrix[0, 2], matrix[1, 2]])) / np.array(
            [matrix[0, 0], matrix[1, 1]]
        )
        radius = np.linalg.norm(xy, axis=2)
        theta = radius.copy()
        for _ in range(8):
            squared = theta**2
            terms = sum(c * squared ** (i + 1) for i, c in enumerate(camera.distortion))
            derivative = sum(
                (3 + 2 * i) * c * squared ** (i + 1)
                for i, c in enumerate(camera.distortion)
            )
            theta -= (theta * (1 + terms) - radius) / (1 + derivative)
        plane = xy * (np.sin(theta) / np.maximum(radius, 1e-12))[:, :, None]
        return np.concatenate((plane, np.cos(theta)[:, :, None]), axis=-1).astype(
            np.float32
        )
    inv = cv2.fisheye.undistortPoints(sensor, camera.matrix, camera.distortion).reshape(
        camera.height * SS, camera.width * SS, 2
    )
    return np.concatenate((inv, np.ones(inv.shape[:2] + (1,))), axis=2).astype(
        np.float32
    )


def board_pose(theta, azimuth, tilt, roll):
    yaw = np.deg2rad(theta)
    az = np.deg2rad(azimuth)
    inc = np.deg2rad(tilt)
    turn = np.deg2rad(roll)
    rad = np.array([np.cos(az), np.sin(az), 0.0])
    axis = np.array([-np.sin(az), np.cos(az), 0.0])
    z = np.array([0.0, 0.0, 1.0])
    center = rad * np.sin(yaw) + z * np.cos(yaw)
    normal = rad * np.sin(inc) + z * np.cos(inc)
    across = rad * np.cos(inc) - z * np.sin(inc)
    return (
        center,
        np.cos(turn) * across + np.sin(turn) * axis,
        -np.sin(turn) * across + np.cos(turn) * axis,
        normal,
    )


def render(camera, spec, ray, grids, rng, negative=False):
    theta, azimuth, tilt, roll, cols, rows, px, lighting = spec
    center, u, v, normal = board_pose(theta, azimuth, tilt, roll)
    side = px / camera.matrix[0, 0]
    pitch = side * np.array([1.43, 1.57])
    extent = np.array([cols, rows]) * pitch
    lattice = rng.choice(len(grids), size=(rows, cols), replace=False)
    spins = rng.integers(0, 4, (rows, cols))
    textures = np.stack(
        [np.rot90(grids[id_], -rot) for id_, rot in zip(lattice.ravel(), spins.ravel())]
    ).reshape(rows, cols, 8, 8)
    mask = np.ones((rows, cols), bool)
    markers = []
    corners = np.array([[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]])
    for j, i in np.ndindex(rows, cols):
        pos = (np.array([i, j]) + 0.5) * pitch - extent * 0.5
        c = center + pos[0] * u + pos[1] * v
        X = c + side * (corners[:, 0, None] * u + corners[:, 1, None] * v)
        q = project(camera, X)
        m = project(camera, (X + np.roll(X, -1, axis=0)) * 0.5)
        # Omit truncated prints from both the renderer and reference.
        if (
            np.min(q) < 4
            or np.max(q[:, 0]) >= camera.width - 5
            or np.max(q[:, 1]) >= camera.height - 5
            or np.min(X[:, 2]) < 0.08
        ):
            mask[j, i] = False
            continue
        a = np.roll(q, -1, axis=0) - q
        b = m - q
        bow = np.abs(a[:, 0] * b[:, 1] - a[:, 1] * b[:, 0]) / np.linalg.norm(a, axis=1)
        if not negative:
            markers.append(
                dict(
                    id=int(lattice[j, i]),
                    corners_px=np.roll(q, -int(spins[j, i]), axis=0).ravel().tolist(),
                    field_angle_deg=float(
                        np.rad2deg(np.arctan2(np.hypot(c[0], c[1]), c[2]))
                    ),
                    maximum_edge_bow_px=float(bow.max()),
                )
            )
    t = (center @ normal) / np.maximum(ray @ normal.astype(np.float32), 1.0e-5)
    # Compute board-plane coordinates from rays rather than homography-warping four projected corners.
    pu = (ray @ u.astype(np.float32)) * t - center @ u + extent[0] / 2
    pv = (ray @ v.astype(np.float32)) * t - center @ v + extent[1] / 2
    col = np.floor(pu / pitch[0]).astype(np.int32)
    row = np.floor(pv / pitch[1]).astype(np.int32)
    inside = (col >= 0) & (row >= 0) & (col < cols) & (row < rows)
    x = (pu - (col + 0.5) * pitch[0]) / side + 0.5
    y = (pv - (row + 0.5) * pitch[1]) / side + 0.5
    col = np.clip(col, 0, cols - 1)
    row = np.clip(row, 0, rows - 1)
    ink = (x >= 0) & (x < 1) & (y >= 0) & (y < 1) & inside & mask[row, col]
    bx = np.clip((8 * x).astype(np.int32), 0, 7)
    by = np.clip((8 * y).astype(np.int32), 0, 7)
    black = textures[row, col, by, bx] == 0
    if negative:
        black = (bx == 0) | (by == 0) | (bx == 7) | (by == 7) | ((bx + by) % 2 == 0)
    yy, xx = np.ogrid[: camera.height * SS, : camera.width * SS]
    bg = 127 + 9 * np.sin(xx / (68 * SS) + yy / (88 * SS)) + 0.014 * xx / SS
    light, dark, blur, noise = (
        (228, 19, 0.40, 1.3),
        (175, 61, 0.65, 2.5),
        (210, 37, 0.75, 2.0),
    )[lighting]
    gain = 1 + 0.07 * (pu / extent[0] - 0.5)
    output = np.where(inside, np.where(ink & black, dark, light) * gain, bg)
    pixels = cv2.resize(
        output.astype(np.float32),
        (camera.width, camera.height),
        interpolation=cv2.INTER_AREA,
    )
    pixels = cv2.GaussianBlur(pixels, (5, 5), blur)
    pixels = np.clip(
        np.rint(pixels + rng.normal(0, noise, pixels.shape)), 0, 255
    ).astype(np.uint8)
    return pixels, markers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("output", type=Path)
    ap.add_argument("--split", choices=["dev", "held"], required=True)
    ap.add_argument("--camera", choices=("aist", "tum-vi"), default="aist")
    a = ap.parse_args()
    camera = AIST if a.camera == "aist" else TUM_VI
    a.output.mkdir(parents=True, exist_ok=False)
    cv2.setNumThreads(1)
    rng = np.random.default_rng(467181 if a.split == "dev" else 194721)
    ray = native_camera_rays(camera)
    grids = grids_from_codebook(CODEBOOK)
    # Off-axis angle, azimuth, plane tilt, in-plane roll, columns, rows and on-axis tag side.
    specs = [
        (0, 0, 6, 5, 15, 9, 38),
        (0, 0, -14, 12, 15, 9, 56),
        (19, 12, 8, -14, 13, 8, 38),
        (34, 165, 18, 9, 11, 8, 33),
        (49, 3, 32, -6, 9, 7, 31),
        (54, 174, 45, 14, 8, 7, 32),
        (48, 158, -8, -11, 12, 8, 29),
        (52, -15, 5, 17, 13, 8, 26),
        (56, 2, 28, 15, 7, 6, 50),
        (37, -161, 8, 7, 9, 6, 62),
    ]
    if camera.wide_rays:
        specs[4:6] = [(63, 35, 51, -6, 9, 7, 31), (67, 144, 57, 14, 8, 7, 32)]
        specs[8:10] = [(71, 40, 62, 15, 7, 6, 43), (60, -154, 28, 7, 9, 6, 51)]
    frames = []
    for i, sp in enumerate(specs):
        for lighting in (0, 1):
            par = list(sp)
            if a.split == "held":
                par[0] = max(0, par[0] + rng.uniform(-4, 4))
                par[1] += rng.uniform(-8, 8)
                par[2] += rng.uniform(-9, 9)
                par[3] += rng.uniform(-8, 8)
                par[6] *= rng.uniform(0.90, 1.12)
            par.append(lighting if i % 3 else lighting * 2)
            name = f"{a.split}-{i:02d}-{lighting}"
            im, markers = render(camera, par, ray, grids, rng)
            path = a.output / (name + ".pgm")
            cv2.imwrite(str(path), im)
            frames.append(
                dict(
                    name=name,
                    path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    markers=markers,
                )
            )
        if i % 3 == 2:
            name = f"{a.split}-{i:02d}-negative"
            im, markers = render(camera, par, ray, grids, rng, True)
            path = a.output / (name + ".pgm")
            cv2.imwrite(str(path), im)
            frames.append(
                dict(
                    name=name,
                    path=path.name,
                    sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                    markers=markers,
                )
            )
    data = dict(
        camera=camera.name,
        calibration_source=camera.source,
        intrinsics=camera.matrix.tolist(),
        distortion=camera.distortion.tolist(),
        size=[camera.width, camera.height],
        supersample=SS,
        scenes=frames,
    )
    (a.output / "manifest.json").write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
