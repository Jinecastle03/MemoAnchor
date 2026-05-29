#!/usr/bin/env python3
"""Best-effort Open3D reconstruction for MemoAnchor RGB-D packages."""

from __future__ import annotations

import argparse
import json
import math
import shutil
import struct
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quaternion_to_matrix(q: dict[str, float]) -> list[list[float]]:
    x = q.get("x", 0.0)
    y = q.get("y", 0.0)
    z = q.get("z", 0.0)
    w = q.get("w", 1.0)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length <= 1e-8:
        return [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    x, y, z, w = x / length, y / length, z / length, w / length
    return [
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ]


def matmul3(a: list[list[float]], b: list[list[float]]) -> list[list[float]]:
    return [[sum(a[r][k] * b[k][c] for k in range(3)) for c in range(3)] for r in range(3)]


def read_confidence_array_if_possible(scan_dir: Path, frame: dict[str, Any]) -> "object | None":
    try:
        import numpy as np
    except ImportError:
        return None

    confidence = frame.get("confidence")
    if not confidence or not confidence.get("planes"):
        return None

    plane = confidence["planes"][0]
    plane_path = scan_dir / "frames" / frame["folder"] / plane["file"]
    if not plane_path.exists():
        return None

    width = int(confidence.get("width", 0))
    height = int(confidence.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    data = plane_path.read_bytes()
    row_stride = int(plane["rowStride"])
    pixel_stride = int(plane["pixelStride"])
    values = []
    for y in range(height):
        row = y * row_stride
        for x in range(width):
            offset = row + x * pixel_stride
            values.append(data[offset] if offset < len(data) else 0)

    return np.array(values, dtype=np.uint8).reshape((height, width))


def read_depth_array_if_possible(scan_dir: Path, frame: dict[str, Any], *, apply_confidence: bool = False) -> "object | None":
    try:
        import numpy as np
    except ImportError:
        return None

    depth = frame.get("depth")
    if not depth or not depth.get("planes"):
        return None

    plane = depth["planes"][0]
    plane_path = scan_dir / "frames" / frame["folder"] / plane["file"]
    if not plane_path.exists():
        return None

    width = int(depth.get("width", 0))
    height = int(depth.get("height", 0))
    if width <= 0 or height <= 0:
        return None

    values = depth_values_from_plane(
        plane_path,
        width,
        height,
        str(depth.get("format", "")),
        int(plane["rowStride"]),
        int(plane["pixelStride"]),
    )
    depth_m = np.array(values, dtype=np.float32).reshape((height, width))
    if not np.isfinite(depth_m).any() or float(np.nanmax(depth_m)) <= 0:
        return None

    if apply_confidence:
        confidence = read_confidence_array_if_possible(scan_dir, frame)
        if confidence is not None:
            if confidence.shape != depth_m.shape:
                from PIL import Image

                confidence = np.asarray(
                    Image.fromarray(confidence).resize((width, height), Image.Resampling.NEAREST),
                    dtype=np.uint8,
                )
            # ARKit confidence is 0=low, 1=medium, 2=high. Low confidence depth is the main source of torn mesh.
            depth_m[confidence <= 0] = 0.0

    return depth_m


def make_extrinsic(frame: dict[str, Any]) -> "object":
    import numpy as np

    rotation_unity = quaternion_to_matrix(frame.get("rotation", {}))
    # Open3D camera coordinates use image-down Y; Unity camera coordinates use up Y.
    flip_y = [[1, 0, 0], [0, -1, 0], [0, 0, 1]]
    rotation = matmul3(rotation_unity, flip_y)
    position = frame.get("position", {})

    camera_to_world = np.eye(4, dtype=np.float64)
    camera_to_world[:3, :3] = np.array(rotation, dtype=np.float64)
    camera_to_world[:3, 3] = np.array(
        [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
        dtype=np.float64,
    )
    return np.linalg.inv(camera_to_world)


def depth_values_from_plane(path: Path, width: int, height: int, fmt: str, row_stride: int, pixel_stride: int) -> list[float]:
    data = path.read_bytes()
    values: list[float] = []
    fmt_lower = fmt.lower()

    for y in range(height):
        row = y * row_stride
        for x in range(width):
            offset = row + x * pixel_stride
            if fmt_lower == "depthfloat32" or fmt_lower == "onecomponent32":
                if offset + 4 > len(data):
                    values.append(0.0)
                else:
                    values.append(struct.unpack_from("<f", data, offset)[0])
            elif fmt_lower == "depthuint16":
                if offset + 2 > len(data):
                    values.append(0.0)
                else:
                    # AR depth uint16 is commonly millimeters. Keep a conservative conversion.
                    values.append(struct.unpack_from("<H", data, offset)[0] / 1000.0)
            else:
                values.append(0.0)

    return values


def write_depth_png_if_possible(scan_dir: Path, frame: dict[str, Any], output_path: Path) -> bool:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        return False

    depth = frame.get("depth")
    if not depth or not depth.get("planes"):
        return False

    plane = depth["planes"][0]
    plane_path = scan_dir / "frames" / frame["folder"] / plane["file"]
    if not plane_path.exists():
        return False

    width = int(depth["width"])
    height = int(depth["height"])
    values = depth_values_from_plane(
        plane_path,
        width,
        height,
        str(depth.get("format", "")),
        int(plane["rowStride"]),
        int(plane["pixelStride"]),
    )
    depth_m = np.array(values, dtype=np.float32).reshape((height, width))
    if not np.isfinite(depth_m).any() or float(np.nanmax(depth_m)) <= 0:
        return False

    confidence = read_confidence_array_if_possible(scan_dir, frame)
    if confidence is not None:
        if confidence.shape != depth_m.shape:
            from PIL import Image

            confidence = np.asarray(
                Image.fromarray(confidence).resize((width, height), Image.Resampling.NEAREST),
                dtype=np.uint8,
            )
        depth_m[confidence <= 0] = 0.0

    depth_m[(depth_m < 0.05) | (depth_m > 8.0) | ~np.isfinite(depth_m)] = 0.0

    depth_mm = np.clip(depth_m * 1000.0, 0, 65535).astype(np.uint16)
    return bool(o3d.io.write_image(str(output_path), o3d.geometry.Image(depth_mm)))


def make_color_image_for_depth(scan_dir: Path, frame: dict[str, Any], width: int, height: int) -> "object | None":
    import numpy as np
    import open3d as o3d
    from PIL import Image

    rgb_file = frame.get("rgbFile")
    if not rgb_file:
        return None

    rgb_path = scan_dir / "frames" / frame["folder"] / rgb_file
    if not rgb_path.exists():
        return None

    image = Image.open(rgb_path).convert("RGB")
    if image.size != (width, height):
        image = image.resize((width, height), Image.Resampling.BILINEAR)

    return o3d.geometry.Image(np.asarray(image, dtype=np.uint8))


def make_blank_color(width: int, height: int) -> "object":
    import numpy as np
    import open3d as o3d

    return o3d.geometry.Image(np.full((height, width, 3), 180, dtype=np.uint8))


def load_color_keyframes(scan_dir: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np
    from PIL import Image

    keyframes: list[dict[str, Any]] = []
    for frame in manifest.get("frames", []):
        if not frame.get("hasIntrinsics") or not frame.get("rgbFile"):
            continue

        rgb_path = scan_dir / "frames" / frame["folder"] / frame["rgbFile"]
        if not rgb_path.exists():
            continue

        image = np.asarray(Image.open(rgb_path).convert("RGB"), dtype=np.float32) / 255.0
        height, width = image.shape[:2]
        focal_length = frame.get("focalLength", {})
        principal_point = frame.get("principalPoint", {})
        position = frame.get("position", {})

        keyframes.append(
            {
                "id": frame.get("id", ""),
                "image": image,
                "width": width,
                "height": height,
                "fx": float(focal_length.get("x", 0.0)),
                "fy": float(focal_length.get("y", 0.0)),
                "cx": float(principal_point.get("x", width / 2)),
                "cy": float(principal_point.get("y", height / 2)),
                "position": np.array(
                    [position.get("x", 0.0), position.get("y", 0.0), position.get("z", 0.0)],
                    dtype=np.float64,
                ),
                "rotation": np.array(quaternion_to_matrix(frame.get("rotation", {})), dtype=np.float64),
                "depth": read_depth_array_if_possible(scan_dir, frame, apply_confidence=True),
            }
        )

    return keyframes


def colorize_mesh_from_keyframes(
    mesh: "object",
    scan_dir: Path,
    manifest: dict[str, Any],
    *,
    use_depth_check: bool = True,
    default_color: tuple[float, float, float] = (0.58, 0.70, 0.74),
) -> int:
    import numpy as np
    import open3d as o3d

    keyframes = load_color_keyframes(scan_dir, manifest)
    if not keyframes:
        return 0

    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    if len(vertices) == 0:
        return 0

    if not mesh.has_vertex_normals():
        mesh.compute_vertex_normals()
    normals = np.asarray(mesh.vertex_normals, dtype=np.float64)

    colors = np.full((len(vertices), 3), default_color, dtype=np.float64)
    best_scores = np.full((len(vertices),), -1e9, dtype=np.float64)

    for keyframe in keyframes:
        rel = vertices - keyframe["position"]
        # For row vectors, rel @ R equals R^T * rel in Unity's camera-local projection.
        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            camera_local = rel @ keyframe["rotation"]
            z = camera_local[:, 2]
        valid = np.isfinite(camera_local).all(axis=1) & (z > 0.05)
        if not np.any(valid):
            continue

        with np.errstate(invalid="ignore", over="ignore", divide="ignore"):
            pixel_x = keyframe["fx"] * (camera_local[:, 0] / z) + keyframe["cx"]
            pixel_y = keyframe["cy"] - keyframe["fy"] * (camera_local[:, 1] / z)

        valid &= np.isfinite(pixel_x) & np.isfinite(pixel_y)
        valid &= pixel_x >= 2
        valid &= pixel_x < keyframe["width"] - 2
        valid &= pixel_y >= 2
        valid &= pixel_y < keyframe["height"] - 2
        if not np.any(valid):
            continue

        depth = keyframe.get("depth") if use_depth_check else None
        if depth is not None:
            depth_h, depth_w = depth.shape
            depth_x = np.zeros_like(pixel_x, dtype=np.int32)
            depth_y = np.zeros_like(pixel_y, dtype=np.int32)
            valid_indices = np.flatnonzero(valid)
            depth_x[valid_indices] = np.clip(
                (pixel_x[valid_indices] / keyframe["width"] * depth_w).astype(np.int32),
                0,
                depth_w - 1,
            )
            depth_y[valid_indices] = np.clip(
                (pixel_y[valid_indices] / keyframe["height"] * depth_h).astype(np.int32),
                0,
                depth_h - 1,
            )
            sampled_depth = depth[depth_y, depth_x]
            depth_diff = np.abs(sampled_depth - z)
            tolerance = np.maximum(0.18, z * 0.10)
            valid &= (sampled_depth <= 0) | (depth_diff <= tolerance)
            if not np.any(valid):
                continue

        u = pixel_x / keyframe["width"]
        v = pixel_y / keyframe["height"]
        center_distance = np.sqrt((u - 0.5) ** 2 + (v - 0.5) ** 2)
        center_score = 1.0 - np.clip(center_distance / 0.7, 0.0, 1.0)
        distance_score = 1.0 / (0.25 + z)

        view = keyframe["position"] - vertices
        view_norm = np.linalg.norm(view, axis=1)
        view_norm = np.maximum(view_norm, 1e-6)
        view_dir = view / view_norm[:, None]
        facing_score = np.clip(np.abs(np.sum(normals * view_dir, axis=1)), 0.0, 1.0)

        score = center_score * 2.2 + distance_score + facing_score * 0.4
        update = valid & (score > best_scores)
        if not np.any(update):
            continue

        sx = np.clip(np.rint(pixel_x[update]).astype(np.int32), 0, keyframe["width"] - 1)
        sy = np.clip(np.rint(pixel_y[update]).astype(np.int32), 0, keyframe["height"] - 1)
        colors[update] = keyframe["image"][sy, sx]
        best_scores[update] = score[update]

    colored_count = int(np.count_nonzero(best_scores > -1e8))
    if colored_count > 0:
        mesh.vertex_colors = o3d.utility.Vector3dVector(colors)
    return colored_count


def cleanup_mesh(mesh: "object") -> "object":
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        return mesh

    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return mesh

    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) == 0:
        return mesh

    triangle_clusters, cluster_triangle_counts, _ = mesh.cluster_connected_triangles()
    if len(cluster_triangle_counts) > 1:
        counts = np.asarray(cluster_triangle_counts)
        largest = int(counts.max())
        keep_threshold = max(160, int(largest * 0.006))
        triangles_to_remove = [
            triangle_index
            for triangle_index, cluster_index in enumerate(triangle_clusters)
            if counts[cluster_index] < keep_threshold
        ]

        if triangles_to_remove and len(triangles_to_remove) < len(mesh.triangles):
            mesh.remove_triangles_by_index(triangles_to_remove)
            mesh.remove_unreferenced_vertices()

    if len(mesh.triangles) > 0:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=2)
        mesh.remove_degenerate_triangles()
        mesh.remove_unreferenced_vertices()

    return mesh


def ordered_boundary_loops(mesh: "object", *, max_loop_vertices: int = 240) -> list[list[int]]:
    import collections
    import numpy as np

    triangles = np.asarray(mesh.triangles, dtype=np.int64)
    edge_counts: dict[tuple[int, int], int] = {}
    for tri in triangles:
        a, b, c = int(tri[0]), int(tri[1]), int(tri[2])
        for u, v in ((a, b), (b, c), (c, a)):
            edge = (u, v) if u < v else (v, u)
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    adjacency: dict[int, set[int]] = collections.defaultdict(set)
    for (u, v), count in edge_counts.items():
        if count == 1:
            adjacency[u].add(v)
            adjacency[v].add(u)

    loops: list[list[int]] = []
    seen_nodes: set[int] = set()
    for start in list(adjacency):
        if start in seen_nodes:
            continue

        stack = [start]
        component: list[int] = []
        seen_nodes.add(start)
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in adjacency[node]:
                if neighbor not in seen_nodes:
                    seen_nodes.add(neighbor)
                    stack.append(neighbor)

        if len(component) < 4 or len(component) > max_loop_vertices:
            continue

        if any(len(adjacency[node]) != 2 for node in component):
            continue

        ordered = [component[0]]
        previous = -1
        current = component[0]
        for _ in range(len(component) + 1):
            neighbors = list(adjacency[current])
            next_node = neighbors[0] if neighbors[0] != previous else neighbors[1]
            if next_node == ordered[0]:
                break
            ordered.append(next_node)
            previous, current = current, next_node

        if len(ordered) == len(component):
            loops.append(ordered)

    return loops


def create_local_hole_fill_mesh(base_mesh: "object") -> "object | None":
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        return None

    base_vertices = np.asarray(base_mesh.vertices, dtype=np.float64)
    base_triangles = np.asarray(base_mesh.triangles, dtype=np.int64)
    if len(base_vertices) == 0 or len(base_triangles) == 0:
        return None

    base_colors = np.asarray(base_mesh.vertex_colors, dtype=np.float64) if base_mesh.has_vertex_colors() else None

    vertices: list[list[float]] = []
    triangles: list[list[int]] = []
    colors: list[list[float]] = []
    filled = 0

    for loop in ordered_boundary_loops(base_mesh):
        points = base_vertices[np.asarray(loop, dtype=np.int64)]
        if not np.isfinite(points).all():
            continue

        centroid = points.mean(axis=0)
        centered = points - centroid
        try:
            _, _, vh = np.linalg.svd(centered, full_matrices=False)
        except np.linalg.LinAlgError:
            continue

        axis_u = vh[0]
        axis_v = vh[1]
        normal = vh[2]
        distances = centered @ normal
        rms = float(np.sqrt(np.mean(distances * distances)))
        projected = np.column_stack((centered @ axis_u, centered @ axis_v))
        shifted = np.roll(projected, -1, axis=0)
        signed_area = 0.5 * float(np.sum(projected[:, 0] * shifted[:, 1] - shifted[:, 0] * projected[:, 1]))
        area = abs(signed_area)
        perimeter = float(np.sum(np.linalg.norm(np.roll(points, -1, axis=0) - points, axis=1)))

        if area < 0.003 or area > 1.6:
            continue
        if perimeter > 6.0:
            continue
        if rms > max(0.035, math.sqrt(area) * 0.04):
            continue

        start = len(vertices)
        vertices.extend(points.tolist())
        vertices.append(centroid.tolist())
        center_index = start + len(loop)

        if base_colors is not None and len(base_colors) == len(base_vertices):
            loop_colors = base_colors[np.asarray(loop, dtype=np.int64)]
            colors.extend(loop_colors.tolist())
            colors.append(loop_colors.mean(axis=0).tolist())

        for i in range(len(loop)):
            a = start + i
            b = start + ((i + 1) % len(loop))
            if signed_area >= 0:
                triangles.append([center_index, a, b])
            else:
                triangles.append([center_index, b, a])

        filled += 1

    if not vertices or not triangles:
        print("local_hole_fill=none")
        return None

    fill_mesh = o3d.geometry.TriangleMesh(
        o3d.utility.Vector3dVector(np.asarray(vertices, dtype=np.float64)),
        o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32)),
    )
    if colors and len(colors) == len(vertices):
        fill_mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(np.asarray(colors, dtype=np.float64), 0.0, 1.0))
    fill_mesh.compute_vertex_normals()
    print(f"local_hole_fill=loops={filled} vertices={len(fill_mesh.vertices)} triangles={len(fill_mesh.triangles)}")
    return fill_mesh


def merge_with_local_hole_fill(mesh: "object") -> "object":
    fill_mesh = create_local_hole_fill_mesh(mesh)
    if fill_mesh is None:
        return mesh

    merged = mesh + fill_mesh
    merged.remove_duplicated_triangles()
    merged.remove_degenerate_triangles()
    merged.remove_unreferenced_vertices()
    merged.compute_vertex_normals()
    return merged


def clamp_vertex_colors(mesh: "object") -> None:
    try:
        import numpy as np
        import open3d as o3d
    except ImportError:
        return

    if not mesh.has_vertex_colors():
        return

    colors = np.asarray(mesh.vertex_colors, dtype=np.float64)
    mesh.vertex_colors = o3d.utility.Vector3dVector(np.clip(colors, 0.0, 1.0))


def try_open3d_tsdf(scan_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> Path | None:
    try:
        import open3d as o3d
    except ImportError:
        return None

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=0.025,
        sdf_trunc=0.08,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )

    integrated = 0
    temp_depth_dir = out_dir / "depth_png"
    temp_depth_dir.mkdir(parents=True, exist_ok=True)

    for frame in manifest.get("frames", []):
        if not frame.get("hasIntrinsics"):
            continue

        depth_png = temp_depth_dir / f"{frame['id']}_depth.png"
        if not write_depth_png_if_possible(scan_dir, frame, depth_png):
            continue

        depth_meta = frame.get("depth", {})
        focal_length = frame.get("focalLength", {})
        principal_point = frame.get("principalPoint", {})

        width = int(depth_meta.get("width", 0))
        height = int(depth_meta.get("height", 0))
        if width <= 0 or height <= 0:
            continue

        # Scale RGB camera intrinsics to the depth image grid for geometry fusion.
        image_resolution = frame.get("imageResolution", {})
        rgb_width = float(image_resolution.get("x", width) or width)
        rgb_height = float(image_resolution.get("y", height) or height)
        scale_x = width / rgb_width
        scale_y = height / rgb_height

        color = make_color_image_for_depth(scan_dir, frame, width, height) or make_blank_color(width, height)
        depth = o3d.io.read_image(str(depth_png))
        intrinsic = o3d.camera.PinholeCameraIntrinsic(
            width,
            height,
            float(focal_length.get("x", 0.0)) * scale_x,
            float(focal_length.get("y", 0.0)) * scale_y,
            float(principal_point.get("x", width / 2)) * scale_x,
            float(principal_point.get("y", height / 2)) * scale_y,
        )
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            color,
            depth,
            depth_scale=1000.0,
            depth_trunc=8.0,
            convert_rgb_to_intensity=False,
        )
        volume.integrate(rgbd, intrinsic, make_extrinsic(frame))
        integrated += 1

    if integrated == 0:
        return None

    mesh = volume.extract_triangle_mesh()
    mesh = cleanup_mesh(mesh)
    mesh.compute_vertex_normals()
    if not mesh.has_vertex_colors():
        colored_count = colorize_mesh_from_keyframes(mesh, scan_dir, manifest)
        if colored_count:
            print(f"colored_vertices={colored_count}/{len(mesh.vertices)}")
    else:
        print(f"integrated_color_vertices={len(mesh.vertices)}")
    mesh = merge_with_local_hole_fill(mesh)
    clamp_vertex_colors(mesh)
    result_path = out_dir / "result_tsdf.ply"
    if not o3d.io.write_triangle_mesh(str(result_path), mesh):
        return None
    return result_path


def find_raw_mesh(scan_dir: Path) -> Path | None:
    raw_mesh = scan_dir / "raw_mesh.obj"
    if raw_mesh.exists():
        return raw_mesh

    nested = list(scan_dir.glob("*/raw_mesh.obj"))
    return nested[0] if nested else None


def colorized_raw_mesh(scan_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> Path | None:
    try:
        import open3d as o3d
    except ImportError:
        return None

    raw_mesh = find_raw_mesh(scan_dir)
    if raw_mesh is None:
        return None

    mesh = o3d.io.read_triangle_mesh(str(raw_mesh), enable_post_processing=True)
    if len(mesh.vertices) == 0 or len(mesh.triangles) == 0:
        return None

    mesh.compute_vertex_normals()
    colored_count = colorize_mesh_from_keyframes(mesh, scan_dir, manifest)
    coverage = colored_count / max(1, len(mesh.vertices))
    print(f"raw_colored_vertices={colored_count}/{len(mesh.vertices)} coverage={coverage:.3f}")
    if colored_count == 0:
        return None

    mesh = cleanup_mesh(mesh)
    mesh = merge_with_local_hole_fill(mesh)
    mesh.compute_vertex_normals()
    clamp_vertex_colors(mesh)

    result = out_dir / "result_raw_colored.ply"
    if not o3d.io.write_triangle_mesh(str(result), mesh):
        return None

    baked = bake_vertex_color_texture(mesh, out_dir)
    if baked is not None:
        return baked

    return result


def choose_result(scan_dir: Path, out_dir: Path, manifest: dict[str, Any]) -> Path | None:
    raw_result = colorized_raw_mesh(scan_dir, out_dir, manifest)
    tsdf_result = try_open3d_tsdf(scan_dir, out_dir, manifest)

    # For MemoAnchor the first goal is a recognizable surface that can receive notes.
    # ARKit's mesh usually preserves room layout better than low-resolution TSDF, while TSDF is a fallback
    # when no raw mesh or too little color projection is available.
    chosen = raw_result or tsdf_result
    if chosen is None:
        return None

    final = out_dir / "result.ply"
    if chosen != final:
        shutil.copy2(chosen, final)
    print(f"chosen_result={chosen.name}")
    return final


def fallback_raw_mesh(scan_dir: Path, out_dir: Path) -> Path | None:
    raw_mesh = find_raw_mesh(scan_dir)
    if raw_mesh is None:
        return None

    result = out_dir / "result.obj"
    shutil.copy2(raw_mesh, result)
    return result

def bake_vertex_color_texture(mesh, out_dir: Path) -> Path | None:
    """
    임시 Texture Baking:
    vertex color를 기반으로 간단한 texture atlas를 만들고 OBJ로 저장한다.

    주의:
    이건 RGB keyframe reprojection 기반의 정석 texture baking은 아니고,
    현재 vertex color 결과를 Unity에서 texture material로 보기 위한 1차 구현이다.
    """

    try:
        import numpy as np
        import cv2
    except ImportError:
        print("bake_vertex_color_texture skipped: numpy or cv2 not installed")
        return None

    vertices = np.asarray(mesh.vertices)
    triangles = np.asarray(mesh.triangles)
    colors = np.asarray(mesh.vertex_colors)

    if len(vertices) == 0 or len(triangles) == 0 or len(colors) == 0:
        return None

    texture_size = 2048
    texture = np.zeros((texture_size, texture_size, 3), dtype=np.uint8)

    obj_path = out_dir / "result_textured.obj"
    mtl_path = out_dir / "result_textured.mtl"
    tex_path = out_dir / "result_texture.png"

    # 아주 단순한 planar UV 생성: XZ 기준
    min_x, min_z = vertices[:, 0].min(), vertices[:, 2].min()
    max_x, max_z = vertices[:, 0].max(), vertices[:, 2].max()

    range_x = max(max_x - min_x, 1e-6)
    range_z = max(max_z - min_z, 1e-6)

    uvs = np.zeros((len(vertices), 2), dtype=np.float32)
    uvs[:, 0] = (vertices[:, 0] - min_x) / range_x
    uvs[:, 1] = (vertices[:, 2] - min_z) / range_z

    # texture에 vertex color 찍기
    for i, uv in enumerate(uvs):
        x = int(np.clip(uv[0] * (texture_size - 1), 0, texture_size - 1))
        y = int(np.clip((1.0 - uv[1]) * (texture_size - 1), 0, texture_size - 1))

        color = np.clip(colors[i] * 255.0, 0, 255).astype(np.uint8)
        texture[y, x] = color[::-1]  # RGB -> BGR for cv2

    # 빈 공간 보간용 dilation
    mask = np.any(texture > 0, axis=2).astype(np.uint8) * 255
    kernel = np.ones((5, 5), np.uint8)

    for _ in range(20):
        dilated = cv2.dilate(texture, kernel, iterations=1)
        empty = mask == 0
        texture[empty] = dilated[empty]
        mask = np.any(texture > 0, axis=2).astype(np.uint8) * 255

    cv2.imwrite(str(tex_path), texture)

    # MTL 저장
    with open(mtl_path, "w", encoding="utf-8") as f:
        f.write("newmtl baked_material\n")
        f.write("Ka 1.000 1.000 1.000\n")
        f.write("Kd 1.000 1.000 1.000\n")
        f.write("Ks 0.000 0.000 0.000\n")
        f.write("d 1.0\n")
        f.write("illum 2\n")
        f.write(f"map_Kd {tex_path.name}\n")

    # OBJ 저장
    with open(obj_path, "w", encoding="utf-8") as f:
        f.write(f"mtllib {mtl_path.name}\n")
        f.write("usemtl baked_material\n")

        for v in vertices:
            f.write(f"v {v[0]} {v[1]} {v[2]}\n")

        for uv in uvs:
            f.write(f"vt {uv[0]} {uv[1]}\n")

        for tri in triangles:
            # OBJ index는 1부터 시작
            a, b, c = tri + 1
            f.write(f"f {a}/{a} {b}/{b} {c}/{c}\n")

    print(f"baked texture obj={obj_path}")
    return obj_path

def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scan", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    scan_dir = Path(args.scan)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = scan_dir / "manifest.json"
    if not manifest_path.exists():
        nested = list(scan_dir.glob("*/manifest.json"))
        manifest_path = nested[0] if nested else manifest_path
    if not manifest_path.exists():
        raise FileNotFoundError(f"manifest.json not found in {scan_dir}")

    manifest = load_json(manifest_path)
    result = choose_result(manifest_path.parent, out_dir, manifest)
    if result is None:
        result = fallback_raw_mesh(manifest_path.parent, out_dir)

    if result is None:
        raise RuntimeError("No result mesh could be generated")

    print(f"result={result}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

