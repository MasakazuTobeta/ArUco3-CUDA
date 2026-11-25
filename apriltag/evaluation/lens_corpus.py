# SPDX-License-Identifier: Apache-2.0
"""Render moving planar tag36h11 targets directly onto a native wide-angle sensor."""
import argparse
from dataclasses import dataclass
import hashlib
import json
import math
from pathlib import Path

import numpy as np

from dense_corpus import CODEBOOK, grids_from_codebook


LENSES = [('pinhole', 100), ('equidistant', 115), ('equidistant', 165)]
EXTERNAL_LENSES = [('equidistant', 190), ('equisolid', 170), ('stereographic', 170)]
SEEDS = {'dev': 291061, 'held': 872407, 'projection-held': 1706903}
UNIT = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]])
FRAMES = 10
DARK, WHITE = 27, 226


@dataclass(frozen=True)
class Camera:
    width: int
    height: int
    model: str
    horizontal_fov_deg: int

    @property
    def focal(self):
        angle = math.radians(self.horizontal_fov_deg / 2)
        if self.model == 'equisolid':
            return .5 * self.width / (2 * math.sin(angle / 2))
        if self.model == 'stereographic':
            return .5 * self.width / (2 * math.tan(angle / 2))
        return .5 * self.width / (angle if self.model == 'equidistant' else math.tan(angle))

    @property
    def principal(self):
        return np.array([(self.width - 1) / 2, (self.height - 1) / 2])

    def project(self, world):
        x, y, z = np.moveaxis(np.asarray(world), -1, 0)
        if self.model != 'pinhole':
            radius = np.hypot(x, y)
            angle = np.arctan2(radius, z)
            if self.model == 'equisolid':
                angle = 2 * np.sin(angle / 2)
            elif self.model == 'stereographic':
                angle = 2 * np.tan(angle / 2)
            scale = np.divide(angle, radius, out=np.zeros_like(angle), where=radius > 0)
        else:
            if np.any(z <= 0):
                raise ValueError('target behind the pinhole')
            scale = 1 / z
        return np.stack([x * scale, y * scale], axis=-1) * self.focal + self.principal

    def rays(self, px, py):
        x = (px - self.principal[0]) / self.focal
        y = (py - self.principal[1]) / self.focal
        radius = np.hypot(x, y)
        if self.model == 'equidistant':
            scale = np.sinc(radius / np.pi)
            return x * scale, y * scale, np.cos(radius)
        if self.model == 'equisolid':
            scale = np.sqrt(1 - (radius / 2)**2)
            return x * scale, y * scale, 1 - radius**2 / 2
        if self.model == 'stereographic':
            scale = 1 / (1 + (radius / 2)**2)
            return x * scale, y * scale, (1 - (radius / 2)**2) * scale
        scale = 1 / np.sqrt(1 + radius**2)
        return x * scale, y * scale, scale

    def description(self):
        def extent(span):
            r = span / (2 * self.focal)
            if self.model == 'equisolid':
                return 4 * math.degrees(math.asin(r / 2))
            if self.model == 'stereographic':
                return 4 * math.degrees(math.atan(r / 2))
            return 2 * math.degrees(r if self.model == 'equidistant' else math.atan(r))
        return dict(model=self.model, horizontal_fov_deg=self.horizontal_fov_deg,
                    vertical_fov_deg=extent(self.height), diagonal_fov_deg=extent(math.hypot(self.width, self.height)),
                    focal_px=self.focal, principal_px=self.principal.tolist(),
                    radial_coefficients=[0, 0, 0, 0] if self.model == 'equidistant' else [])


@dataclass(frozen=True)
class Target:
    id: int
    track: str
    first_frame: int
    center: np.ndarray
    axis_u: np.ndarray
    axis_v: np.ndarray
    side_m: float

    def world(self, uv):
        return self.center + self.side_m * (uv[..., :1] * self.axis_u + uv[..., 1:] * self.axis_v)

    def edges(self, camera, extent=.5, samples=49):
        points = 2 * extent * UNIT
        t = np.linspace(0, 1, samples).reshape(1, -1, 1)
        uv = (1 - t) * points[:, None] + t * np.roll(points, -1, axis=0)[:, None]
        return camera.project(self.world(uv))

    def label(self, camera):
        edges = self.edges(camera)
        q = camera.project(self.world(UNIT))
        chord = np.roll(q, -1, axis=0) - q
        delta = edges - q[:, None]
        signed_distance = (chord[:, 0, None] * delta[:, :, 1] - chord[:, 1, None] * delta[:, :, 0])
        sag = np.abs(signed_distance / np.linalg.norm(chord, axis=1)[:, None]).max(axis=1)
        return dict(track=self.track, id=int(self.id), first_visible_frame=self.first_frame,
                    corners_px=q.ravel().tolist(), edge_max_sag_px=sag.tolist(),
                    center_camera_m=self.center.tolist(), axis_u=self.axis_u.tolist(),
                    axis_v=self.axis_v.tolist(), side_m=self.side_m)


def normalized(value):
    return value / np.linalg.norm(value)


def pose(camera, marker_id, slot, frame, phase, offset, handed):
    # Place targets by their sensor bearing. All camera coordinates below are physical meters,
    # with +X right, +Y down and +Z forward. Each target has a square, planar printed surface.
    centers = ((.48, .39), (.79, .58), (.205, .69))
    sizes = (.128, .238, .15)
    first = (1, 3, 5)[slot]
    x, y = centers[slot]
    x += offset[0] + .021 * math.sin(.37 * frame + phase)
    y += offset[1] + .028 * math.cos(.25 * frame + phase)
    if handed:
        x, y = 1 - x, 1 - y
    bearing = np.array(camera.rays(x * camera.width - .5, y * camera.height - .5))
    tangent = normalized(np.cross([0., 1., 0.], bearing))
    vertical = np.cross(bearing, tangent)
    normal = normalized(bearing + .37 * math.sin(phase + frame * .08) * tangent +
                        .30 * math.cos(phase + frame * .06) * vertical)
    horizontal = normalized(np.cross([0., 1., 0.], normal))
    down = np.cross(normal, horizontal)
    roll = phase + .035 * frame
    u = math.cos(roll) * horizontal + math.sin(roll) * down
    v = -math.sin(roll) * horizontal + math.cos(roll) * down
    # Equal apparent sizes across lens models keep the rectilinear control useful.
    axes = np.stack([u, v])
    derivatives = (camera.project(bearing + 1e-4 * axes) - camera.project(bearing - 1e-4 * axes)) / 2e-4
    side = (.12, .21, .15)[slot]
    distance = side * np.mean(np.linalg.norm(derivatives, axis=1)) / (camera.height * sizes[slot])
    return Target(int(marker_id), ('central', 'peripheral-a', 'peripheral-b')[slot], first,
                  bearing * distance, u, v, side)


def background(camera, phase):
    y, x = np.ogrid[:camera.height, :camera.width]
    x = x * (720 / camera.height)
    y = y * (720 / camera.height)
    return np.rint(140 + 12 * np.sin(.027 * x + .014 * y + phase)
                   + 9 * np.cos(.013 * x - .022 * y) + 6 * np.cos(.017 * y + phase)).astype(np.uint8)


def draw(frame, camera, target, grid, samples, full_sensor=False):
    # Sample the projected quiet-zone perimeter for a conservative working rectangle.
    # The inverse ray-plane intersection below supplies all edge and cell membership.
    outline = target.edges(camera, .625, 129).reshape(-1, 2)
    if np.any(outline < 1.5) or np.any(outline > (camera.width - 2.5, camera.height - 2.5)):
        raise ValueError('clipped quiet zone in a fully visible clip')
    x0, y0 = np.floor(outline.min(axis=0) - 2).astype(int)
    x1, y1 = np.ceil(outline.max(axis=0) + 2).astype(int) + 1
    if full_sensor:
        x0, y0, x1, y1 = 0, 0, camera.width, camera.height
    patch = frame[y0:y1, x0:x1]
    base = patch.copy()
    result = np.zeros(patch.shape, np.uint16)
    y, x = np.ogrid[y0:y1, x0:x1]
    normal = np.cross(target.axis_u, target.axis_v)
    plane = np.dot(target.center, normal)
    offset_u = np.dot(target.center, target.axis_u)
    offset_v = np.dot(target.center, target.axis_v)
    positions = (np.arange(samples) + .5) / samples - .5
    for dy in positions:
        for dx in positions:
            rays = camera.rays(x + dx, y + dy)
            def dot(axis):
                return rays[0] * axis[0] + rays[1] * axis[1] + rays[2] * axis[2]
            denominator = dot(normal)
            distance = np.divide(plane, denominator, out=np.zeros_like(denominator), where=denominator > 1e-10)
            u = (distance * dot(target.axis_u) - offset_u) / target.side_m + .5
            v = (distance * dot(target.axis_v) - offset_v) / target.side_m + .5
            on_paper = (distance > 0) & (u >= -.125) & (u < 1.125) & (v >= -.125) & (v < 1.125)
            inside = (u >= 0) & (u < 1) & (v >= 0) & (v < 1)
            i = np.floor(np.clip(u * 8, 0, 7)).astype(np.intp)
            j = np.floor(np.clip(v * 8, 0, 7)).astype(np.intp)
            ink = inside & (grid[j, i] == 0)
            result += np.where(on_paper, np.where(ink, DARK, WHITE), base).astype(np.uint16)
    patch[:] = (result + samples**2 // 2) // samples**2
    return np.array([x0, y0, x1, y1])


def generate(output, split, height, projection_check=False):
    if projection_check and split != 'projection-held':
        raise ValueError('the projection cross-check uses the projection-held ID pool')
    seed = 2351671 if projection_check else SEEDS[split]
    output.mkdir(parents=True, exist_ok=True)
    width = height * 16 // 9
    grids = grids_from_codebook(CODEBOOK)
    rng = np.random.default_rng(seed)
    # Disjoint ID pools make the development and held splits independently useful.
    start = 0 if split == 'dev' else len(grids) // 2
    stop = len(grids) // 2 if split == 'dev' else len(grids)
    pool = np.arange(start, stop)
    if split == 'projection-held':
        occupied = [np.random.default_rng(SEEDS[k]).choice(subset, 9, replace=False)
                    for k, subset in [('dev', np.arange(len(grids) // 2)),
                                      ('held', np.arange(len(grids) // 2, len(grids)))]]
        pool = np.setdiff1d(np.arange(len(grids)), np.concatenate(occupied))
    ids = rng.choice(pool, (3, 3), replace=False)
    phases = rng.uniform(-1.1, 1.1, (3, 3))
    offsets = rng.uniform(-.014, .014, (3, 3, 2))
    scenes, models = [], {}
    lenses = EXTERNAL_LENSES if split == 'projection-held' else LENSES
    if projection_check:
        lenses = [('equisolid', 155), ('equidistant', 178), ('stereographic', 165)]
    for index, (model, fov) in enumerate(lenses):
        camera = Camera(width, height, model, fov)
        name = f'{model}-{fov}'
        models[name] = camera.description()
        field = background(camera, phases[index, 0])
        for samples in (1, 4 if split == 'projection-held' else 3):
            clip = f'{name}-ss{samples}'
            for f in range(FRAMES):
                frame = field.copy()
                markers, bounds, absent = [], [], []
                for slot, (first, last) in enumerate(((1, 6), (3, 8), (5, 7))):
                    if not first <= f <= last:
                        absent.append(int(ids[index, slot]))
                        continue
                    target = pose(camera, ids[index, slot], slot, f, phases[index, slot],
                                  offsets[index, slot], bool((index + (split == 'held')) % 2))
                    box = draw(frame, camera, target, grids[target.id], samples)
                    if any(np.all(np.minimum(box[2:], b[2:]) > np.maximum(box[:2], b[:2])) for b in bounds):
                        raise ValueError('overlapping targets require depth compositing')
                    bounds.append(box)
                    markers.append(target.label(camera))
                stem = f'{clip}-{f:02d}'
                data = f'P5\n{width} {height}\n255\n'.encode() + frame.tobytes()
                (output / (stem + '.pgm')).write_bytes(data)
                scenes.append(dict(name=stem, path=stem + '.pgm', sha256=hashlib.sha256(data).hexdigest(),
                                   clip=clip, frame_index=f, camera=name, supersample=samples,
                                   markers=markers, partial_markers=[], must_be_absent_ids=absent))
    manifest = dict(dictionary='tag36h11', split=split, seed=seed, width=width, height=height,
                    numpy_version=np.__version__, codebook_sha256=hashlib.sha256(CODEBOOK.read_bytes()).hexdigest(),
                    cameras=models, dark_gray8=DARK, white_gray8=WHITE, nominal_fps=30,
                    corner_convention='canonical TL, TR, BR, BL; native distorted sensor pixels; integer pixel centers',
                    scenes=scenes)
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--split', choices=SEEDS, required=True)
    parser.add_argument('--height', type=int, choices=(720, 1080, 2160), required=True)
    parser.add_argument('--projection-check', action='store_true',
                        help='use the independent 155/178/165-degree projection schedule')
    args = parser.parse_args()
    generate(args.output, args.split, args.height, args.projection_check)
