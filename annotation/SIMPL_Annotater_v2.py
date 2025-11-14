#!/usr/bin/env python3
"""Interactive SAM-based building refinement with chip-level prompts."""

from __future__ import annotations

import argparse
import datetime as dt
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
import threading
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import rasterio
from rasterio import features
from rasterio.mask import mask as rio_mask
from rasterio.transform import Affine, array_bounds
from rasterio.warp import transform_geom
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon, box, mapping, shape
from shapely.ops import unary_union

import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image, ImageDraw, ImageTk

import torch
import torch.nn.functional as F
from ultralytics import SAM as UltralyticsSAM


ColorPoint = Tuple[float, float]


def _now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _expand_polygon(geom: Polygon | MultiPolygon | GeometryCollection) -> Polygon:
    if geom.is_empty:
        raise ValueError("Geometry is empty.")
    if isinstance(geom, Polygon):
        return geom
    if isinstance(geom, MultiPolygon):
        return max(geom.geoms, key=lambda g: g.area)
    if isinstance(geom, GeometryCollection):
        polys = [g for g in geom.geoms if isinstance(g, (Polygon, MultiPolygon))]
        if not polys:
            raise ValueError("GeometryCollection contains no polygons.")
        merged = unary_union(polys)
        if isinstance(merged, (MultiPolygon, GeometryCollection)):
            return _expand_polygon(merged)
        return merged
    raise TypeError(f"Unsupported geometry type: {geom.geom_type}")


def _clip_value(value: float, min_value: float, max_value: float) -> float:
    return float(max(min_value, min(max_value, value)))


def polygon_rings_to_pixel_coords(polygon: Polygon, affine: Affine) -> Tuple[List[ColorPoint], List[List[ColorPoint]]]:
    inv = ~affine

    def to_pix(seq) -> List[ColorPoint]:
        coords: List[ColorPoint] = []
        for x, y in seq:
            col, row = inv * (x, y)
            coords.append((float(col), float(row)))
        return coords

    exterior = to_pix(polygon.exterior.coords)
    interiors = [to_pix(ring.coords) for ring in polygon.interiors]
    return exterior, interiors


def _geometry_to_pixel_coords(polygon: Polygon, affine: Affine) -> List[ColorPoint]:
    exterior, _ = polygon_rings_to_pixel_coords(polygon, affine)
    return exterior


def _sample_polygon_boundary(
    polygon: Polygon,
    affine: Affine,
    target_points: int,
    clamp_width: int,
    clamp_height: int,
) -> List[ColorPoint]:
    inv = ~affine
    points: List[ColorPoint] = []
    if polygon.length == 0 or target_points <= 0:
        return points
    for fraction in np.linspace(0.0, 1.0, num=target_points, endpoint=False):
        pt = polygon.exterior.interpolate(fraction, normalized=True)
        col, row = inv * (pt.x, pt.y)
        points.append(
            (
                _clip_value(col, 0, clamp_width - 1),
                _clip_value(row, 0, clamp_height - 1),
            )
        )
    return points


def _buffer_ring_points(
    polygon: Polygon,
    affine: Affine,
    num_points: int,
    buffer_pixels: float,
    clamp_width: int,
    clamp_height: int,
) -> List[ColorPoint]:
    if num_points <= 0:
        return []
    buffer_distance = buffer_pixels * max(abs(affine.a), abs(affine.e))
    ring = polygon.buffer(buffer_distance)
    if ring.is_empty:
        return []
    outer = _expand_polygon(ring)
    boundary = outer.exterior
    inv = ~affine
    points = []
    for fraction in np.linspace(0.0, 1.0, num=num_points, endpoint=False):
        pt = boundary.interpolate(fraction, normalized=True)
        col, row = inv * (pt.x, pt.y)
        points.append(
            (
                _clip_value(col, 0, clamp_width - 1),
                _clip_value(row, 0, clamp_height - 1),
            )
        )
    return points


def _pad_image_to_stride(image: np.ndarray, stride: int) -> Tuple[np.ndarray, int, int]:
    height, width = image.shape[:2]
    pad_bottom = (stride - (height % stride)) % stride
    pad_right = (stride - (width % stride)) % stride
    if pad_bottom == 0 and pad_right == 0:
        return image, 0, 0
    padded = np.pad(image, ((0, pad_bottom), (0, pad_right), (0, 0)), mode="edge")
    return padded, pad_bottom, pad_right


def _sanitize_filename(text: str) -> str:
    import re

    text = text.strip() or "annotation"
    return re.sub(r"[^0-9A-Za-z._-]+", "_", text)


@dataclass
class BuildingMask:
    idx: int
    feature_id: str
    properties: Dict[str, Any]
    polygon: Polygon
    pixel_polygon: List[ColorPoint]
    pixel_polygon_interiors: List[List[ColorPoint]]
    bbox_row_min: float
    bbox_row_max: float
    bbox_col_min: float
    bbox_col_max: float
    refined_mask: Optional[np.ndarray] = None
    clicks_positive: List[ColorPoint] = field(default_factory=list)
    clicks_negative: List[ColorPoint] = field(default_factory=list)
    accepted: bool = False
    skipped: bool = False
    deleted: bool = False


@dataclass
class ChipData:
    array: np.ndarray
    display: np.ndarray
    transform: Affine
    row_off: int
    col_off: int
    polygon_chip: List[ColorPoint]
    polygon_holes_chip: List[List[ColorPoint]]
    initial_mask: np.ndarray
    rough_centroid: Optional[ColorPoint]
    rough_bbox: Optional[Tuple[float, float, float, float]]


class ImageSession:
    def __init__(
        self,
        image_path: Path,
        annotations_path: Path,
        annotation_crs: Optional[str],
        display_bands: Optional[Sequence[int]],
    ) -> None:
        self.image_path = Path(image_path).expanduser().resolve()
        self.annotations_path = Path(annotations_path).expanduser().resolve()

        with rasterio.open(self.image_path) as src:
            self.crs = src.crs
            base_profile = src.profile.copy()

            dataset_mask = src.dataset_mask()
            valid_shapes = []
            if dataset_mask is not None:
                for geom, val in features.shapes(dataset_mask, transform=src.transform):
                    if val != 0:
                        valid_shapes.append(shape(geom))

            if valid_shapes:
                valid_geom = unary_union(valid_shapes).buffer(0)
            else:
                valid_geom = box(*src.bounds)

            if valid_geom.is_empty:
                valid_geom = box(*src.bounds)

            clipped, clipped_transform = rio_mask(src, [mapping(valid_geom)], crop=True)

            mask_array: Optional[np.ndarray] = None
            data = clipped
            if isinstance(clipped, np.ma.MaskedArray):
                mask_array = np.ma.getmaskarray(clipped)
                data = clipped.data

            valid_mask: Optional[np.ndarray] = None
            if mask_array is not None and mask_array is not np.ma.nomask:
                valid_mask = ~np.all(mask_array, axis=0)
            else:
                nodata = base_profile.get("nodata")
                if nodata is not None:
                    valid_mask = np.any(data != nodata, axis=0)

            if valid_mask is not None:
                if not valid_mask.any():
                    raise ValueError(f"No valid pixels found within {self.image_path.name}.")
                row_indices = np.where(valid_mask.any(axis=1))[0]
                col_indices = np.where(valid_mask.any(axis=0))[0]
                row_start, row_end = int(row_indices[0]), int(row_indices[-1]) + 1
                col_start, col_end = int(col_indices[0]), int(col_indices[-1]) + 1
                data = data[:, row_start:row_end, col_start:col_end]
                if mask_array is not None and mask_array is not np.ma.nomask:
                    mask_array = mask_array[:, row_start:row_end, col_start:col_end]
                clipped_transform = clipped_transform * Affine.translation(col_start, row_start)

        if mask_array is not None and mask_array is not np.ma.nomask:
            trimmed = np.ma.array(data, mask=mask_array).filled(0)
        else:
            trimmed = np.asarray(data)

        self.image_array = trimmed
        self.band_count = trimmed.shape[0]
        self.height = trimmed.shape[1]
        self.width = trimmed.shape[2]
        self.transform = clipped_transform
        self.valid_polygon = valid_geom
        self.bounds_polygon = box(*array_bounds(self.height, self.width, self.transform))

        self.profile = base_profile
        self.profile.update({"height": self.height, "width": self.width, "transform": self.transform})

        self.display_rgb = self._build_display_array(clipped, display_bands)

        self.buildings: List[BuildingMask] = self._load_buildings(annotation_crs)

    @property
    def stem(self) -> str:
        return self.image_path.stem

    def close(self) -> None:
        pass

    def _build_display_array(
        self,
        clipped_array: np.ndarray,
        display_bands: Optional[Sequence[int]],
    ) -> np.ndarray:
        if display_bands is None:
            if self.band_count >= 3:
                display_bands = (1, 2, 3)
            elif self.band_count == 2:
                display_bands = (1, 2, 1)
            else:
                display_bands = (1, 1, 1)

        bands = []
        for band_idx in display_bands:
            band_idx = max(1, min(self.band_count, band_idx))
            band = np.asarray(clipped_array[band_idx - 1])
            bands.append(band)

        stacked = np.stack(bands, axis=-1)
        if stacked.dtype != np.uint8:
            stacked = np.clip(stacked, 0, 255).astype(np.uint8)
        return np.ascontiguousarray(stacked)

    def _infer_geojson_crs(self, data: Dict[str, Any], fallback: Optional[str]) -> str:
        if "crs" in data:
            crs_obj = data["crs"]
            if isinstance(crs_obj, dict):
                props = crs_obj.get("properties", {})
                if "name" in props:
                    return props["name"]
                if "crs" in props:
                    return props["crs"]
        if fallback:
            return fallback
        return "EPSG:4326"

    def _load_buildings(self, annotation_crs: Optional[str]) -> List[BuildingMask]:
        with open(self.annotations_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        src_crs = self._infer_geojson_crs(data, annotation_crs)
        dst_crs = self.crs.to_string() if self.crs else src_crs

        buildings: List[BuildingMask] = []
        for idx, feature in enumerate(data.get("features", [])):
            geom = feature.get("geometry")
            if not geom:
                continue

            transformed = transform_geom(src_crs, dst_crs, geom, precision=6)
            try:
                polygon = _expand_polygon(shape(transformed).buffer(0))
            except (ValueError, TypeError):
                continue

            if not polygon.intersects(self.valid_polygon):
                continue
            polygon = polygon.intersection(self.valid_polygon)
            if polygon.is_empty:
                continue
            if isinstance(polygon, (MultiPolygon, GeometryCollection)):
                try:
                    polygon = _expand_polygon(polygon)
                except (ValueError, TypeError):
                    continue

            pixel_exterior, pixel_interiors = polygon_rings_to_pixel_coords(polygon, self.transform)
            pixel_polygon = pixel_exterior
            props = feature.get("properties", {}).copy()
            cols = [p[0] for p in pixel_polygon]
            rows = [p[1] for p in pixel_polygon]
            bbox_col_min = max(0.0, min(cols))
            bbox_col_max = min(float(self.width - 1), max(cols))
            bbox_row_min = max(0.0, min(rows))
            bbox_row_max = min(float(self.height - 1), max(rows))

            feature_id = str(
                feature.get("id")
                or props.get("id")
                or props.get("building_id")
                or f"polygon_{idx:04d}"
            )

            buildings.append(
                BuildingMask(
                    idx=len(buildings),
                    feature_id=feature_id,
                    properties=props,
                    polygon=polygon,
                    pixel_polygon=pixel_polygon,
                    pixel_polygon_interiors=pixel_interiors,
                    bbox_row_min=bbox_row_min,
                    bbox_row_max=bbox_row_max,
                    bbox_col_min=bbox_col_min,
                    bbox_col_max=bbox_col_max,
                )
            )

        return buildings

    def get_chip(self, building: BuildingMask, buffer: int = 96) -> ChipData:
        row_min = max(0, int(math.floor(building.bbox_row_min) - buffer))
        row_max = min(self.height, int(math.ceil(building.bbox_row_max) + buffer))
        col_min = max(0, int(math.floor(building.bbox_col_min) - buffer))
        col_max = min(self.width, int(math.ceil(building.bbox_col_max) + buffer))

        if row_max <= row_min or col_max <= col_min:
            raise ValueError("Invalid chip bounds computed for building.")

        window = rasterio.windows.Window(col_min, row_min, col_max - col_min, row_max - row_min)
        transform = rasterio.windows.transform(window, self.transform)

        chip_array = self.image_array[:, row_min:row_max, col_min:col_max]
        chip_display = self.display_rgb[row_min:row_max, col_min:col_max].copy()
        chip_height, chip_width = chip_display.shape[:2]

        initial_mask = features.rasterize(
            [(building.polygon, 1)],
            out_shape=(chip_height, chip_width),
            transform=transform,
            fill=0,
            dtype=np.uint8,
        )

        def clamp_point(col: float, row: float) -> ColorPoint:
            return (
                _clip_value(col, 0, chip_width - 1),
                _clip_value(row, 0, chip_height - 1),
            )

        polygon_chip = [clamp_point(col - col_min, row - row_min) for col, row in building.pixel_polygon]
        polygon_holes_chip = [
            [clamp_point(col - col_min, row - row_min) for col, row in ring]
            for ring in building.pixel_polygon_interiors
        ]

        rough_centroid: Optional[ColorPoint] = None
        rough_bbox: Optional[Tuple[float, float, float, float]] = None
        if initial_mask is not None and initial_mask.any():
            rows, cols = np.nonzero(initial_mask)
            if len(rows) > 0:
                centroid_col = float(cols.mean())
                centroid_row = float(rows.mean())
                rough_centroid = clamp_point(centroid_col, centroid_row)
                bbox_col_min = float(cols.min())
                bbox_row_min = float(rows.min())
                bbox_col_max = float(cols.max())
                bbox_row_max = float(rows.max())
                rough_bbox = (
                    _clip_value(bbox_col_min, 0, chip_width - 1),
                    _clip_value(bbox_row_min, 0, chip_height - 1),
                    _clip_value(bbox_col_max, 0, chip_width - 1),
                    _clip_value(bbox_row_max, 0, chip_height - 1),
                )

        return ChipData(
            array=chip_array,
            display=chip_display,
            transform=transform,
            row_off=row_min,
            col_off=col_min,
            polygon_chip=polygon_chip,
            polygon_holes_chip=polygon_holes_chip,
            initial_mask=initial_mask,
            rough_centroid=rough_centroid,
            rough_bbox=rough_bbox,
        )

    def rasterize_building(self, building: BuildingMask) -> np.ndarray:
        mask = features.rasterize(
            [(building.polygon, 1)],
            out_shape=(self.height, self.width),
            transform=self.transform,
            fill=0,
            dtype=np.uint8,
        )
        return mask


class SamSegmenter:
    def __init__(self, model_path: str, device: Optional[str] = None, imgsz: int = 1024) -> None:
        self._pad_bottom = 0
        self._pad_right = 0
        self._orig_shape: Optional[Tuple[int, int]] = None
        self.supports_box = True

        try:
            sam_model = UltralyticsSAM(model_path)
        except Exception as exc:  # pragma: no cover - surface a clearer error
            raise RuntimeError(f"Failed to load SAM weights from '{model_path}': {exc}") from exc

        is_sam2 = bool(getattr(sam_model, "is_sam2", False))
        if not is_sam2:
            raise ValueError(
                "This tool now only supports SAM 2 checkpoints. "
                f"Provided weights '{model_path}' are not identified as SAM 2."
            )

        if isinstance(imgsz, (tuple, list)):
            imgsz_override = tuple(int(v) for v in imgsz)
        else:
            imgsz_override = (int(imgsz), int(imgsz))

        overrides = {
            "model": model_path,
            "imgsz": imgsz_override,
            "save": False,
            "task": "segment",
            "mode": "predict",
            "batch": 1,
        }
        if device:
            overrides["device"] = device

        predictor_cls = sam_model._smart_load("predictor")
        self.predictor = predictor_cls(overrides=overrides, _callbacks=sam_model.callbacks)
        self.predictor.setup_model(model=sam_model.model, verbose=False)

    def set_image(self, image_rgb: np.ndarray) -> None:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("SAM expects an RGB image with shape (H, W, 3).")
        self._orig_shape = image_rgb.shape[:2]
        padded, pad_bottom, pad_right = _pad_image_to_stride(image_rgb, stride=32)
        self._pad_bottom = pad_bottom
        self._pad_right = pad_right
        bgr = padded[:, :, ::-1]
        self.predictor.set_image(bgr)

    def predict(
        self,
        point_coords: Optional[np.ndarray] = None,
        point_labels: Optional[np.ndarray] = None,
        box: Optional[np.ndarray] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = False,
    ) -> Optional[np.ndarray]:
        has_points = point_coords is not None and np.asarray(point_coords).size > 0
        has_masks = mask_input is not None and np.asarray(mask_input).size > 0
        has_box = box is not None

        if not (has_points or has_masks or has_box):
            raise ValueError("At least one prompt (points, box, or mask) is required for SAM prediction.")

        kwargs: Dict[str, Any] = {"multimask_output": multimask_output}

        if has_points:
            coords = np.asarray(point_coords, dtype=np.float32)
            if coords.ndim == 1:
                coords = coords[None, :]

            if point_labels is None:
                labels = np.ones(coords.shape[0], dtype=np.int32)
            else:
                labels = np.asarray(point_labels, dtype=np.int32)
            if labels.ndim == 0:
                labels = labels[None]

            if labels.shape[0] != coords.shape[0]:
                raise ValueError("Point labels must align with the provided point coordinates.")

            kwargs["points"] = np.expand_dims(coords, axis=0)
            kwargs["labels"] = np.expand_dims(labels, axis=0)

        if has_masks:
            mask_arr = np.asarray(mask_input, dtype=np.float32)
            if mask_arr.ndim == 2:
                # single mask -> add N dimension
                mask_arr = mask_arr[None, :, :]  # (1, H, W)
            elif mask_arr.ndim == 3:
                # already (N, H, W), good
                pass
            else:
                raise ValueError("Mask prompt must have shape (H, W) or (N, H, W).")

            # Keep as float32 (0/1 or 0..1), which matches result.masks.data
            kwargs["masks"] = mask_arr

        if box is not None and self.supports_box:
            bbox_arr = np.asarray(box, dtype=np.float32)
            kwargs["bboxes"] = bbox_arr[None, :] if bbox_arr.ndim == 1 else bbox_arr

        results = self.predictor(**kwargs)
        if not results:
            return None

        result = results[0]
        if result.masks is None or not len(result.masks.data):
            return None

        mask = result.masks.data[0].cpu().numpy().astype(np.uint8)
        if self._pad_bottom or self._pad_right:
            h, w = self._orig_shape or mask.shape
            mask = mask[:h, :w]
        return mask


class OutputManager:
    def __init__(self, session: ImageSession, output_dir: Path, save_masks: bool, autosave_every: int) -> None:
        self.session = session
        self.output_dir = Path(output_dir).expanduser().resolve()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.save_masks = save_masks
        self.autosave_every = max(1, autosave_every)

        stem = session.stem
        self.geojson_path = self.output_dir / f"{stem}_refined.geojson"
        self.audit_path = self.output_dir / f"{stem}_audit.json"
        self.features: List[Dict[str, Any]] = []
        self.audit_log: List[Dict[str, Any]] = []
        self._pending_since_flush = 0
        self.deleted_ids: set[str] = set()

    def record(self, building: BuildingMask, mask: np.ndarray, timestamp: str) -> None:
        geometry = self._mask_to_geometry(mask)
        if geometry is None:
            messagebox.showwarning("Mask empty", "SAM returned an empty mask; nothing saved.")
            return

        building_id = building.feature_id
        self.deleted_ids.discard(building_id)
        self._remove_feature(building_id)

        feature = {
            "type": "Feature",
            "geometry": geometry,
            "properties": self._build_properties(building, mask, timestamp),
        }
        self.features.append(feature)

        self.audit_log.append(
            {
                "timestamp": timestamp,
                "image_path": str(self.session.image_path),
                "building_id": building_id,
                "action": "accept",
                "clicks": {
                    "positive": [list(map(float, pt)) for pt in building.clicks_positive],
                    "negative": [list(map(float, pt)) for pt in building.clicks_negative],
                },
                "mask_area_pixels": int(mask.sum()),
            }
        )

        if self.save_masks:
            self._write_mask_tif(building, mask)

        self._pending_since_flush += 1
        if self._pending_since_flush >= self.autosave_every:
            self.flush()

    def record_deletion(self, building: BuildingMask, timestamp: str) -> None:
        building_id = building.feature_id
        self.deleted_ids.add(building_id)
        removed = self._remove_feature(building_id)
        if removed and self.save_masks:
            fname = f"{self.session.stem}_{_sanitize_filename(building_id)}_mask.tif"
            try:
                (self.output_dir / fname).unlink(missing_ok=True)
            except Exception:
                pass

        self.audit_log.append(
            {
                "timestamp": timestamp,
                "image_path": str(self.session.image_path),
                "building_id": building_id,
                "action": "delete",
                "reason": "marked_missing",
                "mask_area_pixels": 0,
            }
        )
        self._pending_since_flush += 1
        if self._pending_since_flush >= self.autosave_every:
            self.flush()

    def flush(self) -> None:
        feature_collection = {"type": "FeatureCollection", "features": self.features}
        with open(self.geojson_path, "w", encoding="utf-8") as f:
            json.dump(feature_collection, f, indent=2)

        with open(self.audit_path, "w", encoding="utf-8") as f:
            json.dump(self.audit_log, f, indent=2)

        self._pending_since_flush = 0

    def _remove_feature(self, building_id: str) -> bool:
        before = len(self.features)
        self.features = [
            feat for feat in self.features if feat.get("properties", {}).get("building_id") != building_id
        ]
        return before != len(self.features)

    def _mask_to_geometry(self, mask: np.ndarray) -> Optional[Dict[str, Any]]:
        if mask.sum() == 0:
            return None
        shapes_iter = features.shapes(mask.astype(np.uint8), transform=self.session.transform)
        polygons = []
        for geom, value in shapes_iter:
            if value == 0:
                continue
            polygon = shape(geom).buffer(0)
            if polygon.area > 0:
                polygons.append(polygon)
        if not polygons:
            return None
        merged = unary_union(polygons)
        return mapping(merged)

    def _write_mask_tif(self, building: BuildingMask, mask: np.ndarray) -> None:
        fname = f"{self.session.stem}_{_sanitize_filename(building.feature_id)}_mask.tif"
        out_path = self.output_dir / fname
        profile = self.session.profile.copy()
        profile.update(driver="GTiff", count=1, dtype=rasterio.uint8, compress="lzw")
        with rasterio.open(out_path, "w", **profile) as dst:
            dst.write(mask.astype(np.uint8), 1)

    def _build_properties(self, building: BuildingMask, mask: np.ndarray, timestamp: str) -> Dict[str, Any]:
        props = building.properties.copy()
        props.update(
            {
                "building_id": building.feature_id,
                "source": "overture",
                "refined_by": "sam+human",
                "timestamp": timestamp,
                "mask_sum_pixels": int(mask.sum()),
                "clicks_positive": [list(map(float, pt)) for pt in building.clicks_positive],
                "clicks_negative": [list(map(float, pt)) for pt in building.clicks_negative],
                "pixel_polygon": [list(map(float, pt)) for pt in building.pixel_polygon],
            }
        )
        return props


class SamAnnotatorApp:
    def __init__(self, entries: List[Tuple[Path, Path]], args: argparse.Namespace, device: str) -> None:
        self.entries = entries
        self.total_images = len(entries)
        self.args = args
        self.device = device
        self.segmenter = SamSegmenter(model_path=str(args.sam_checkpoint), device=device)
        self.buffer_px = args.chip_buffer
        prompt_mode = getattr(args, "prompt_mode", "points")
        self.prompt_mode = prompt_mode if prompt_mode in {"points", "box", "mask"} else "points"

        self.entry_index: int = -1
        self.session: Optional[ImageSession] = None
        self.output_manager: Optional[OutputManager] = None

        self.current_building: Optional[BuildingMask] = None
        self.current_chip: Optional[ChipData] = None
        self.current_mask_chip: Optional[np.ndarray] = None
        self.current_chip_rgb: Optional[np.ndarray] = None

        self.zoom: float = 1.0
        self.min_zoom: float = 0.25
        self.max_zoom: float = 6.0
        self.canvas_image_ref: Optional[ImageTk.PhotoImage] = None
        self.undo_stack: List[Tuple[str, ColorPoint]] = []
        self.sam_running = False
        self.show_mask: bool = True
        self.delete_mode = False
        self.paint_mode: Optional[str] = None

        self.root = tk.Tk()
        self.root.title("SAM Building Refiner")
        self.root.protocol("WM_DELETE_WINDOW", self.finish_and_exit)

        self.mode_var = tk.StringVar(value="positive")
        self.status_var = tk.StringVar(value="Load an image to begin.")

        self._build_ui()
        self._bind_events()
        self.load_session(0)

    # UI -----------------------------------------------------------------
    def _build_ui(self) -> None:
        self.root.columnconfigure(1, weight=1)
        self.root.rowconfigure(0, weight=1)

        sidebar = ttk.Frame(self.root, padding=8)
        sidebar.grid(row=0, column=0, sticky="ns")
        sidebar.rowconfigure(1, weight=1)

        ttk.Label(sidebar, text="Buildings").grid(row=0, column=0, sticky="ew")
        self.building_list = tk.Listbox(sidebar, height=25, exportselection=False)
        self.building_list.grid(row=1, column=0, sticky="nsew", pady=(4, 8))

        nav_frame = ttk.Frame(sidebar)
        nav_frame.grid(row=2, column=0, sticky="ew", pady=6)
        ttk.Button(nav_frame, text="Prev", command=self.prev_building).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(nav_frame, text="Next", command=self.next_building).grid(row=0, column=1)

        ttk.Label(sidebar, textvariable=self.status_var, wraplength=220, foreground="#444").grid(
            row=3, column=0, sticky="ew", pady=(6, 0)
        )

        canvas_frame = ttk.Frame(self.root, padding=8)
        canvas_frame.grid(row=0, column=1, sticky="nsew")
        canvas_frame.rowconfigure(0, weight=1)
        canvas_frame.columnconfigure(0, weight=1)

        self.canvas = tk.Canvas(canvas_frame, background="#111", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        toolbar = ttk.Frame(self.root, padding=(8, 4))
        toolbar.grid(row=1, column=0, columnspan=2, sticky="ew")
        for col in range(10):
            toolbar.columnconfigure(col, weight=0)
        toolbar.columnconfigure(9, weight=1)

        ttk.Radiobutton(toolbar, text="Positive (+)", variable=self.mode_var, value="positive").grid(row=0, column=0)
        ttk.Radiobutton(toolbar, text="Negative (-)", variable=self.mode_var, value="negative").grid(row=0, column=1)
        self.undo_btn = ttk.Button(toolbar, text="Undo (z)", command=self.undo_last)
        self.undo_btn.grid(row=0, column=2, padx=(12, 0))
        self.reset_btn = ttk.Button(toolbar, text="Reset clicks", command=self.reset_clicks)
        self.reset_btn.grid(row=0, column=3, padx=(6, 0))
        self.resegment_btn = ttk.Button(toolbar, text="Re-segment (a)", command=lambda: self.run_sam_async())
        self.resegment_btn.grid(row=0, column=4, padx=(12, 0))
        self.accept_btn = ttk.Button(toolbar, text="Accept (s)", command=self.accept_mask)
        self.accept_btn.grid(row=0, column=5, padx=(12, 0))
        self.skip_btn = ttk.Button(toolbar, text="Skip (k)", command=self.skip_building)
        self.skip_btn.grid(row=0, column=6, padx=(12, 0))
        self.save_image_btn = ttk.Button(toolbar, text="Save & Next Image", command=self.save_and_next_image)
        self.save_image_btn.grid(row=0, column=7, padx=(12, 0))

        self.toggle_mask_btn = ttk.Button(toolbar, text="Toggle Mask (m)", command=self.toggle_mask)
        self.toggle_mask_btn.grid(row=1, column=0, padx=(12, 0), pady=(6, 0))
        self.neighbor_accept_btn = ttk.Button(toolbar, text="Accept Nearby", command=self.accept_neighbor_masks)
        self.neighbor_accept_btn.grid(row=1, column=1, padx=(12, 0), pady=(6, 0))
        self.delete_btn = ttk.Button(toolbar, text="Delete (d)", command=self.start_delete_mode)
        self.delete_btn.grid(row=1, column=2, padx=(12, 0), pady=(6, 0))

        self.brush_size = tk.IntVar(value=6)
        self.brush_btn = ttk.Button(toolbar, text="Brush", command=lambda: self.start_paint_mode("brush"))
        self.brush_btn.grid(row=1, column=3, padx=(12, 0), pady=(6, 0))
        self.eraser_btn = ttk.Button(toolbar, text="Eraser", command=lambda: self.start_paint_mode("eraser"))
        self.eraser_btn.grid(row=1, column=4, padx=(4, 0), pady=(6, 0))
        ttk.Label(toolbar, text="Size").grid(row=1, column=5, sticky="e", padx=(8, 2), pady=(6, 0))
        self.brush_spin = tk.Spinbox(toolbar, from_=1, to=64, width=4, textvariable=self.brush_size)
        self.brush_spin.grid(row=1, column=6, sticky="w", pady=(6, 0))
        self.finish_brush_btn = ttk.Button(toolbar, text="Done", command=self.finish_paint_mode)
        self.finish_brush_btn.grid(row=1, column=7, padx=(8, 0), pady=(6, 0))
        self.exit_btn = ttk.Button(toolbar, text="Exit", command=self.finish_and_exit)
        self.exit_btn.grid(row=1, column=8, padx=(12, 0), pady=(6, 0))

        ttk.Label(
            toolbar,
            text="Scroll/trackpad = zoom, Right/Middle drag = pan, Left-click adds points.",
            foreground="#555",
        ).grid(row=1, column=9, sticky="e", pady=(6, 0))

        self.action_buttons = [
            self.resegment_btn,
            self.accept_btn,
            self.skip_btn,
            self.save_image_btn,
            self.neighbor_accept_btn,
            self.delete_btn,
            self.brush_btn,
            self.eraser_btn,
            self.finish_brush_btn,
        ]

    def _bind_events(self) -> None:
        self.building_list.bind("<<ListboxSelect>>", self.on_list_select)

        self.canvas.bind("<Button-1>", self.on_canvas_click)
        self.canvas.bind("<B1-Motion>", self.on_canvas_drag)
        self.canvas.bind("<ButtonPress-2>", self.on_pan_start)
        self.canvas.bind("<B2-Motion>", self.on_pan_move)
        self.canvas.bind("<ButtonPress-3>", self.on_pan_start)
        self.canvas.bind("<B3-Motion>", self.on_pan_move)
        self.canvas.bind("<MouseWheel>", self.on_mouse_wheel)
        self.canvas.bind("<Button-4>", self.on_mouse_wheel)
        self.canvas.bind("<Button-5>", self.on_mouse_wheel)

        self.root.bind("a", lambda _: self.run_sam_async())
        self.root.bind("s", lambda _: self.accept_mask())
        self.root.bind("k", lambda _: self.skip_building())
        self.root.bind("n", lambda _: self.next_building())
        self.root.bind("p", lambda _: self.prev_building())
        self.root.bind("z", lambda _: self.undo_last())
        self.root.bind("+", lambda _: self.adjust_zoom(1.2))
        self.root.bind("=", lambda _: self.adjust_zoom(1.2))
        self.root.bind("-", lambda _: self.adjust_zoom(0.8))
        self.root.bind("_", lambda _: self.adjust_zoom(0.8))
        self.root.bind("m", lambda _: self.toggle_mask())
        self.root.bind("d", lambda _: self.start_delete_mode())

    def set_busy(self, busy: bool) -> None:
        state = "disabled" if busy else "normal"
        cursor = "watch" if busy else ""
        for btn in self.action_buttons:
            btn.state([state] if busy else ["!disabled"])
        self.root.configure(cursor=cursor)
        self.canvas.configure(cursor="watch" if busy else "")
        self.sam_running = busy
        if busy:
            self._cancel_delete_mode()
            self.finish_paint_mode()

    def _status_prefix(self) -> str:
        image_idx = self.entry_index + 1 if self.entry_index >= 0 else 0
        prefix = f"Image {image_idx}/{self.total_images}" if self.total_images else "Image"
        if self.session and self.current_building:
            accepted = sum(1 for b in self.session.buildings if b.accepted)
            prefix += (
                f" • Building {self.current_building.idx + 1}/{len(self.session.buildings)}"
                f" • Accepted {accepted}"
            )
        return prefix

    def set_status(self, message: str) -> None:
        self.status_var.set(f"{self._status_prefix()} — {message}")

    def adjust_zoom(self, factor: float) -> None:
        if not self.current_chip:
            return
        x_view = self.canvas.xview()[0] if self.canvas_image_ref else 0
        y_view = self.canvas.yview()[0] if self.canvas_image_ref else 0
        self.zoom = _clip_value(self.zoom * factor, self.min_zoom, self.max_zoom)
        self.render_canvas()
        if self.canvas_image_ref:
            self.canvas.xview_moveto(x_view)
            self.canvas.yview_moveto(y_view)

    def toggle_mask(self) -> None:
        self.show_mask = not self.show_mask
        state = "shown" if self.show_mask else "hidden"
        self.set_status(f"Mask overlay {state}.")
        self.render_canvas()

    def start_paint_mode(self, mode: str) -> None:
        if not self.current_chip or self.sam_running:
            self.set_status("Load a building and finish any SAM run before painting.")
            return
        if self.current_mask_chip is None and self.current_chip is not None:
            self.current_mask_chip = self.current_chip.initial_mask.copy()
        if self.current_mask_chip is None:
            self.set_status("No mask available to edit.")
            return
        self.finish_paint_mode()
        self.paint_mode = mode
        self._cancel_delete_mode()
        cursor = "dotbox" if mode == "brush" else "circle"
        self.canvas.configure(cursor=cursor)
        label = "Brush" if mode == "brush" else "Eraser"
        self.set_status(f"{label} mode — click or drag to edit the mask. Click Done when finished.")

    def finish_paint_mode(self) -> None:
        if self.paint_mode:
            self.paint_mode = None
            self.canvas.configure(cursor="")

    def start_delete_mode(self) -> None:
        if not self.current_building or not self.current_chip:
            self.set_status("Select a building before deleting.")
            return
        if self.sam_running:
            self.set_status("Wait for the current SAM run to finish before deleting.")
            return
        self.finish_paint_mode()
        if self.current_building.deleted:
            self.set_status("Building already deleted.")
            return
        mask = self.current_mask_chip if self.current_mask_chip is not None else self.current_chip.initial_mask
        if mask is None or not mask.any():
            self.set_status("No mask pixels available for deletion.")
            return
        self.delete_mode = True
        self.canvas.configure(cursor="X_cursor")
        self.set_status("Delete mode — click inside the building to remove it.")

    def _cancel_delete_mode(self) -> None:
        if self.delete_mode:
            self.delete_mode = False
            self.canvas.configure(cursor="")

    def _handle_delete_click(self, chip_x: float, chip_y: float) -> None:
        self._cancel_delete_mode()
        if not self.current_building or not self.current_chip:
            return
        removed = self._erase_component_at(int(round(chip_x)), int(round(chip_y)))
        if not removed:
            self.set_status("No mask pixels at that location; deletion cancelled.")
            return
        if self.current_chip:
            self.current_chip.initial_mask = self.current_mask_chip.copy()
        if self.current_mask_chip.sum() == 0:
            self.current_building.deleted = True
            self.current_building.accepted = False
            self.current_building.skipped = False
            self.current_building.clicks_positive.clear()
            self.current_building.clicks_negative.clear()
            self.undo_stack.clear()
            global_mask = np.zeros((self.session.height, self.session.width), dtype=np.uint8)
            self.current_building.refined_mask = global_mask
            if self.output_manager is not None:
                self.output_manager.record_deletion(self.current_building, _now_iso())
            self._update_building_entry(self.current_building.idx, "✗")
            self.set_status(f"Deleted {self.current_building.feature_id}.")
        else:
            remaining = int(self.current_mask_chip.sum())
            self.set_status(
                f"Removed component; {remaining} pixels remain. Delete the rest or refine as needed."
            )
        self.render_canvas()

    def _erase_component_at(self, col: int, row: int) -> bool:
        if self.current_mask_chip is None:
            return False
        mask = self.current_mask_chip
        h, w = mask.shape
        if not (0 <= row < h and 0 <= col < w):
            return False
        if mask[row, col] == 0:
            return False
        stack = [(row, col)]
        while stack:
            r, c = stack.pop()
            if mask[r, c] == 0:
                continue
            mask[r, c] = 0
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                if 0 <= nr < h and 0 <= nc < w and mask[nr, nc] != 0:
                    stack.append((nr, nc))
        self.current_mask_chip = mask
        return True

    def _apply_brush(self, chip_x: float, chip_y: float) -> None:
        if not self.paint_mode or not self.current_chip:
            return
        if self.current_mask_chip is None:
            base = self.current_chip.initial_mask
            if base is None:
                return
            self.current_mask_chip = base.copy()
        mask = self.current_mask_chip
        radius = max(1, int(self.brush_size.get()))
        row = int(round(chip_y))
        col = int(round(chip_x))
        h, w = mask.shape
        r0 = max(0, row - radius)
        r1 = min(h, row + radius + 1)
        c0 = max(0, col - radius)
        c1 = min(w, col + radius + 1)
        if r0 >= r1 or c0 >= c1:
            return
        yy, xx = np.ogrid[r0:r1, c0:c1]
        circle = (yy - row) ** 2 + (xx - col) ** 2 <= radius**2
        patch = mask[r0:r1, c0:c1]
        if self.paint_mode == "brush":
            patch[circle] = 1
        else:
            patch[circle] = 0
        mask[r0:r1, c0:c1] = patch
        self.current_mask_chip = mask
        action = "Added" if self.paint_mode == "brush" else "Erased"
        self.set_status(f"{action} pixels; mask now has {int(mask.sum())} pixels.")
        self.render_canvas()

    def _neighbors_in_chip(self, include_current: bool = False) -> List[BuildingMask]:
        if not self.session or not self.current_chip:
            return []
        neighbors: List[BuildingMask] = []
        chip = self.current_chip
        col_off = chip.col_off
        row_off = chip.row_off
        col_max = col_off + chip.display.shape[1]
        row_max = row_off + chip.display.shape[0]
        for building in self.session.buildings:
            if not include_current and building is self.current_building:
                continue
            if building.bbox_col_max < col_off or building.bbox_col_min > col_max:
                continue
            if building.bbox_row_max < row_off or building.bbox_row_min > row_max:
                continue
            neighbors.append(building)
        return neighbors

    # Session management -------------------------------------------------
    def load_session(self, index: int) -> None:
        if index < 0 or index >= self.total_images:
            messagebox.showinfo("Done", "All images processed.")
            self.finish_and_exit()
            return

        if self.output_manager is not None:
            self.output_manager.flush()
        if self.session is not None:
            self.session.close()

        image_path, annotations_path = self.entries[index]
        self.session = ImageSession(
            image_path=image_path,
            annotations_path=annotations_path,
            annotation_crs=self.args.annotation_crs,
            display_bands=self.args.display_bands,
        )

        if not self.session.buildings:
            messagebox.showinfo("No buildings", f"No overlapping polygons for {image_path.name}. Skipping.")
            self.load_session(index + 1)
            return

        self.output_manager = OutputManager(
            session=self.session,
            output_dir=self.args.output_dir,
            save_masks=self.args.save_mask_tif,
            autosave_every=self.args.autosave_every,
        )

        self.entry_index = index
        self.root.title(f"SAM Building Refiner — {image_path.name}")
        self._populate_building_list()
        self.zoom = 1.0
        self.canvas.delete("all")
        self.canvas.configure(scrollregion=(0, 0, self.session.display_rgb.shape[1], self.session.display_rgb.shape[0]))
        self.set_status("Select a building to begin.")
        self.goto_building(0)

    def _populate_building_list(self) -> None:
        self.building_list.delete(0, tk.END)
        if not self.session:
            return
        for building in self.session.buildings:
            if building.deleted:
                marker = "✗"
            elif building.accepted:
                marker = "✓"
            elif building.skipped:
                marker = "–"
            else:
                marker = " "
            self.building_list.insert(tk.END, f"[{marker}] {building.feature_id}")

    def goto_building(self, idx: int) -> None:
        if not self.session:
            return
        if idx >= len(self.session.buildings):
            self.save_and_next_image()
            return
        idx = max(0, idx)
        self.building_list.selection_clear(0, tk.END)
        self.building_list.selection_set(idx)
        self.building_list.see(idx)
        self.on_list_select()

    def finish_and_exit(self) -> None:
        if self.output_manager is not None:
            self.output_manager.flush()
        if self.session is not None:
            self.session.close()
        self.root.destroy()

    def run(self) -> None:
        self.root.mainloop()

    # Building / chip handling ------------------------------------------
    def on_list_select(self, event: Optional[tk.Event] = None) -> None:
        if not self.session:
            return
        selection = self.building_list.curselection()
        if not selection:
            return
        idx = selection[0]
        building = self.session.buildings[idx]
        self.current_building = building
        self.undo_stack.clear()
        self._cancel_delete_mode()
        self.finish_paint_mode()

        try:
            chip = self.session.get_chip(building, buffer=self.buffer_px)
        except ValueError as exc:
            self.set_status(f"Unable to extract chip: {exc}")
            return

        self.current_chip = chip
        self.current_chip_rgb = np.ascontiguousarray(chip.display.astype(np.uint8))

        if building.refined_mask is not None:
            r0, c0 = chip.row_off, chip.col_off
            h, w = chip.display.shape[:2]
            self.current_mask_chip = building.refined_mask[r0 : r0 + h, c0 : c0 + w].copy()
        else:
            self.current_mask_chip = chip.initial_mask.copy()

        self.zoom = self._auto_zoom(chip.display.shape[:2])
        self.render_canvas()
        self.set_status("Press s to accept original mask or a to refine with SAM.")

    def _auto_zoom(self, shape: Tuple[int, int]) -> float:
        height, width = shape
        largest = max(height, width)
        if largest <= 0:
            return 1.0
        target = 512.0
        return _clip_value(max(1.0, target / largest), self.min_zoom, self.max_zoom)

    def render_canvas(self) -> None:
        if not self.current_chip or not self.current_building:
            return

        base = Image.fromarray(self.current_chip.display.astype(np.uint8)).convert("RGBA")

        if self.show_mask and self.current_mask_chip is not None:
            mask_img = Image.fromarray((self.current_mask_chip > 0).astype(np.uint8) * 255, mode="L")
            mask_overlay = Image.new("RGBA", base.size, (255, 0, 0, 130))
            base = Image.composite(mask_overlay, base, mask_img)

        overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        poly = self.current_chip.polygon_chip
        if poly and len(poly) >= 2:
            draw.line(poly + [poly[0]], fill=(0, 176, 255, 220), width=2)
        for ring_pts in self.current_chip.polygon_holes_chip:
            if len(ring_pts) >= 2:
                draw.line(ring_pts, fill=(0, 176, 255, 160), width=1)
        for other in self._neighbors_in_chip(include_current=False):
            if other.accepted:
                outline_color = (120, 255, 120, 160)
            else:
                outline_color = (180, 180, 255, 120)
            pts = [
                (
                    _clip_value(col - self.current_chip.col_off, 0, self.current_chip.display.shape[1] - 1),
                    _clip_value(row - self.current_chip.row_off, 0, self.current_chip.display.shape[0] - 1),
                )
                for col, row in other.pixel_polygon
            ]
            if len(pts) >= 2:
                draw.line(pts + [pts[0]], fill=outline_color, width=1)

        for col, row in self._points_to_chip(self.current_building.clicks_positive):
            radius = 2
            draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=(0, 255, 0, 230))
        for col, row in self._points_to_chip(self.current_building.clicks_negative):
            radius = 2
            draw.ellipse((col - radius, row - radius, col + radius, row + radius), fill=(255, 165, 0, 230))

        base = Image.alpha_composite(base, overlay)

        zoomed_size = (
            max(1, int(base.width * self.zoom)),
            max(1, int(base.height * self.zoom)),
        )
        display_img = base.resize(zoomed_size, resample=Image.NEAREST)
        self.canvas_image_ref = ImageTk.PhotoImage(display_img)
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.canvas_image_ref)
        self.canvas.configure(scrollregion=(0, 0, zoomed_size[0], zoomed_size[1]))

    # Canvas interactions ------------------------------------------------
    def on_canvas_click(self, event: tk.Event) -> None:
        if not self.current_chip or not self.current_building:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        chip_x = canvas_x / self.zoom
        chip_y = canvas_y / self.zoom

        chip_width = self.current_chip.display.shape[1]
        chip_height = self.current_chip.display.shape[0]
        if chip_x < 0 or chip_y < 0 or chip_x >= chip_width or chip_y >= chip_height:
            return

        if self.delete_mode:
            self._handle_delete_click(chip_x, chip_y)
            return
        if self.paint_mode:
            self._apply_brush(chip_x, chip_y)
            return

        global_col = chip_x + self.current_chip.col_off
        global_row = chip_y + self.current_chip.row_off
        point = (
            _clip_value(global_col, 0, self.session.width - 1),
            _clip_value(global_row, 0, self.session.height - 1),
        )

        if self.mode_var.get() == "positive":
            self.current_building.clicks_positive.append(point)
            self.undo_stack.append(("positive", point))
        else:
            self.current_building.clicks_negative.append(point)
            self.undo_stack.append(("negative", point))

        self.render_canvas()
        self.set_status("Added prompt; press a to refine or s to accept.")

    def on_canvas_drag(self, event: tk.Event) -> None:
        if not self.paint_mode or not self.current_chip or not self.current_building:
            return
        canvas_x = self.canvas.canvasx(event.x)
        canvas_y = self.canvas.canvasy(event.y)
        chip_x = canvas_x / self.zoom
        chip_y = canvas_y / self.zoom
        chip_width = self.current_chip.display.shape[1]
        chip_height = self.current_chip.display.shape[0]
        if chip_x < 0 or chip_y < 0 or chip_x >= chip_width or chip_y >= chip_height:
            return
        self._apply_brush(chip_x, chip_y)

    def on_pan_start(self, event: tk.Event) -> None:
        self.canvas.scan_mark(event.x, event.y)

    def on_pan_move(self, event: tk.Event) -> None:
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    def on_mouse_wheel(self, event: tk.Event) -> None:
        if not self.current_chip:
            return
        factor = 1.2 if (event.num == 4 or getattr(event, "delta", 0) > 0) else 0.8
        self.adjust_zoom(factor)

    # SAM inference ------------------------------------------------------
    def run_sam_async(self, initial: bool = False, auto_message: bool = True) -> None:
        if not self.current_building or not self.current_chip:
            return
        if self.sam_running:
            self.set_status("SAM already running…")
            return

        building = self.current_building
        chip = self.current_chip
        chip_height, chip_width = self.current_chip.display.shape[:2]
        if building.deleted:
            self.set_status("Cannot refine a building that has been deleted.")
            return

        positive_chip: List[ColorPoint] = []
        negative_chip: List[ColorPoint] = []
        if self.prompt_mode == "points" and chip.rough_centroid:
            positive_chip.append(chip.rough_centroid)
        positive_chip.extend(self._points_to_chip(building.clicks_positive))
        negative_chip.extend(self._points_to_chip(building.clicks_negative))

        coords: Optional[np.ndarray] = None
        labels: Optional[np.ndarray] = None
        if positive_chip or negative_chip:
            coords = np.array(positive_chip + negative_chip, dtype=np.float32)
            coords[:, 0] = np.clip(coords[:, 0], 0, chip_width - 1)
            coords[:, 1] = np.clip(coords[:, 1], 0, chip_height - 1)
            labels = np.concatenate(
                (
                    np.ones(len(positive_chip), dtype=np.int32),
                    np.zeros(len(negative_chip), dtype=np.int32),
                )
            )

        mask_prompt: Optional[np.ndarray] = None
        if self.prompt_mode == "mask":
            initial_mask = chip.initial_mask
            if initial_mask is not None and initial_mask.any():
                mask_prompt = np.ascontiguousarray(initial_mask.astype(np.uint8))

        bbox: Optional[np.ndarray] = None
        if self.prompt_mode == "box" and chip.rough_bbox:
            x0, y0, x1, y1 = chip.rough_bbox
            pad = 2.0
            bbox = np.array(
                [
                    max(0.0, x0 - pad),
                    max(0.0, y0 - pad),
                    min(float(chip_width - 1), x1 + pad),
                    min(float(chip_height - 1), y1 + pad),
                ],
                dtype=np.float32,
            )
            if bbox[2] <= bbox[0]:
                bbox[2] = min(float(chip_width - 1), bbox[0] + 4.0)
            if bbox[3] <= bbox[1]:
                bbox[3] = min(float(chip_height - 1), bbox[1] + 4.0)

        if coords is None and mask_prompt is None and bbox is None:
            self.set_status("No prompts available — add clicks or choose a different prompt mode.")
            return

        chip_rgb = self.current_chip_rgb
        if chip_rgb is None:
            chip_rgb = np.ascontiguousarray(chip.display.astype(np.uint8))
            self.current_chip_rgb = chip_rgb
        chip_shape = chip.display.shape[:2]

        self.sam_running = True
        self.set_busy(True)
        self.set_status("Running SAM refinement…")

        def work() -> None:
            try:
                self.segmenter.set_image(chip_rgb)
                mask = self.segmenter.predict(
                    point_coords=coords,
                    point_labels=labels,
                    box=bbox,
                    mask_input=mask_prompt,
                )
            except Exception as exc:  # pragma: no cover - UI only
                import traceback

                traceback.print_exc()
                err_msg = str(exc) if str(exc) else exc.__class__.__name__
                def report_error() -> None:
                    self.sam_running = False
                    self.set_busy(False)
                    self.set_status(f"SAM error: {err_msg}")

                self.root.after(0, report_error)
                return

            if mask is not None and mask.shape != chip_shape:
                def mismatch() -> None:
                    self.sam_running = False
                    self.set_busy(False)
                    self.set_status(
                        f"Mask size mismatch ({mask.shape} vs chip {chip_shape}); retry."
                    )

                self.root.after(0, mismatch)
                return

            def finish() -> None:
                current_building = self.current_building
                self.sam_running = False
                if current_building is not building:
                    self.set_busy(False)
                    return
                self.set_busy(False)
                if mask is None or mask.sum() == 0:
                    self.set_status("SAM returned an empty mask; add more clicks and retry.")
                    return
                self.current_mask_chip = (mask > 0).astype(np.uint8)
                if auto_message:
                    action = "Initial segmentation" if initial else "Mask updated"
                    self.set_status(f"{action}; review and press s to accept.")
                self.render_canvas()

            self.root.after(0, finish)

        threading.Thread(target=work, daemon=True).start()

    def _points_to_chip(self, points: Sequence[ColorPoint]) -> List[ColorPoint]:
        if not self.current_chip:
            return []
        chip_points = []
        width = self.current_chip.display.shape[1]
        height = self.current_chip.display.shape[0]
        for col, row in points:
            chip_points.append(
                (
                    _clip_value(col - self.current_chip.col_off, 0, width - 1),
                    _clip_value(row - self.current_chip.row_off, 0, height - 1),
                )
            )
        return chip_points

    # Point management ---------------------------------------------------
    def undo_last(self) -> None:
        if not self.current_building or not self.undo_stack:
            return
        mode, point = self.undo_stack.pop()
        if mode == "positive":
            try:
                self.current_building.clicks_positive.remove(point)
            except ValueError:
                pass
        else:
            try:
                self.current_building.clicks_negative.remove(point)
            except ValueError:
                pass
        self.render_canvas()
        self.set_status("Removed last prompt point.")

    def reset_clicks(self) -> None:
        if not self.current_building or not self.current_chip:
            return
        self.current_building.clicks_positive.clear()
        self.current_building.clicks_negative.clear()
        self.undo_stack.clear()
        if self.current_building.refined_mask is not None:
            r0, c0 = self.current_chip.row_off, self.current_chip.col_off
            h, w = self.current_chip.display.shape[:2]
            self.current_mask_chip = self.current_building.refined_mask[r0 : r0 + h, c0 : c0 + w].copy()
        else:
            self.current_mask_chip = self.current_chip.initial_mask.copy()
        self.set_status("Clicks cleared. Re-segment if needed.")
        self.render_canvas()

    # Building navigation ------------------------------------------------
    def prev_building(self) -> None:
        selection = self.building_list.curselection()
        if not selection:
            return
        self.goto_building(selection[0] - 1)

    def next_building(self) -> None:
        selection = self.building_list.curselection()
        if not selection:
            return
        self.goto_building(selection[0] + 1)

    def _update_building_entry(self, idx: int, status: str) -> None:
        building = self.session.buildings[idx]
        self.building_list.delete(idx)
        self.building_list.insert(idx, f"[{status}] {building.feature_id}")

    # Accept / skip ------------------------------------------------------
    def accept_mask(self) -> None:
        if not self.session or not self.current_building or self.current_mask_chip is None or not self.current_chip:
            messagebox.showinfo("Nothing to save", "Use the existing polygon or run SAM before accepting.")
            return
        if self.current_building.deleted:
            self.set_status("Building marked as deleted; no mask to accept.")
            return
        if self.current_building.accepted:
            self.set_status("Already saved for this building.")
            return

        chip_mask = (self.current_mask_chip > 0).astype(np.uint8)
        global_mask = np.zeros((self.session.height, self.session.width), dtype=np.uint8)
        r0, c0 = self.current_chip.row_off, self.current_chip.col_off
        h, w = chip_mask.shape
        global_mask[r0 : r0 + h, c0 : c0 + w] = chip_mask

        self.current_building.refined_mask = global_mask
        self.current_building.accepted = True
        self.current_building.skipped = False
        timestamp = _now_iso()
        self.output_manager.record(self.current_building, global_mask, timestamp)
        self._update_building_entry(self.current_building.idx, "✓")

        next_idx = self.current_building.idx + 1
        self.set_status(f"Saved mask for {self.current_building.feature_id}.")
        self.goto_building(next_idx)

    def skip_building(self) -> None:
        if not self.current_building:
            return
        if self.current_building.deleted:
            self.set_status(f"{self.current_building.feature_id} already deleted.")
            self.goto_building(self.current_building.idx + 1)
            return
        self.current_building.skipped = True
        self.current_building.accepted = False
        self._update_building_entry(self.current_building.idx, "–")
        self.set_status(f"Skipped {self.current_building.feature_id}.")
        self.goto_building(self.current_building.idx + 1)

    def save_and_next_image(self) -> None:
        if not self.session:
            return
        self.set_status("Saving results...")
        self.output_manager.flush()
        self.session.close()
        next_index = self.entry_index + 1
        if next_index >= self.total_images:
            messagebox.showinfo("Done", "All images processed.")
            self.finish_and_exit()
        else:
            self.load_session(next_index)

    def accept_neighbor_masks(self) -> None:
        if not self.session or not self.current_chip:
            return
        if self.sam_running:
            self.set_status("Finish current SAM run before accepting nearby buildings.")
            return
        neighbors = self._neighbors_in_chip(include_current=False)
        accepted_count = 0
        for building in neighbors:
            if building.accepted or building.deleted:
                continue
            mask = self.session.rasterize_building(building)
            building.refined_mask = mask
            building.accepted = True
            building.skipped = False
            self.output_manager.record(building, mask, _now_iso())
            self._update_building_entry(building.idx, "✓")
            accepted_count += 1
        if accepted_count:
            self.set_status(f"Accepted {accepted_count} nearby building(s).")
            self.render_canvas()
        else:
            self.set_status("No nearby buildings accepted.")


def load_manifest(path: Path) -> List[Tuple[Path, Path]]:
    path = Path(path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    if path.suffix.lower() == ".json":
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return [(Path(item["image"]), Path(item["annotations"])) for item in data]
    if path.suffix.lower() in {".csv", ".txt"}:
        import csv

        entries: List[Tuple[Path, Path]] = []
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                entries.append((Path(row["image"]), Path(row["annotations"])))
        return entries
    raise ValueError(f"Unsupported manifest extension: {path.suffix}")


def discover_city_entries(
    images_dir: Path,
    masks_dir: Path,
    cities: Optional[Sequence[str]] = None,
) -> List[Tuple[Path, Path]]:
    images_dir = Path(images_dir).expanduser().resolve()
    masks_dir = Path(masks_dir).expanduser().resolve()
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")
    if not masks_dir.exists():
        raise FileNotFoundError(f"Masks directory not found: {masks_dir}")

    if cities:
        candidate_cities = [str(city) for city in cities]
    else:
        candidate_cities = sorted([p.name for p in images_dir.iterdir() if p.is_dir()])
    if not candidate_cities:
        raise ValueError(f"No city subdirectories found in {images_dir}")

    entries: List[Tuple[Path, Path]] = []
    for city in candidate_cities:
        city_dir = images_dir / city
        if not city_dir.is_dir():
            print(f"[warn] City directory missing: {city_dir}")
            continue

        mask_path = masks_dir / f"{city}_buildings.geojson"
        if not mask_path.exists():
            matches = list(masks_dir.glob(f"*{city}*_buildings.geojson"))
            if matches:
                mask_path = matches[0]
            else:
                print(f"[warn] No GeoJSON mask for city '{city}' in {masks_dir}; skipping.")
                continue

        image_paths = [
            path
            for path in city_dir.rglob("*")
            if path.is_file() and path.suffix.lower() in {".tif", ".tiff"}
        ]
        if not image_paths:
            print(f"[warn] No GeoTIFF images found for city '{city}'.")
            continue

        print(f"[info] Found {len(image_paths)} images for {city}; using mask {mask_path.name}.")
        for image_path in sorted(image_paths):
            entries.append((image_path, mask_path))

    if not entries:
        raise ValueError("No image/annotation pairs discovered. Verify directory structure and naming.")
    return entries


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM-assisted building mask refiner.")
    parser.add_argument("--image", type=Path, help="Path to GeoTIFF image.")
    parser.add_argument("--annotations", type=Path, help="Path to Overture GeoJSON.")
    parser.add_argument("--batch-manifest", type=Path, help="JSON or CSV manifest with image/annotation pairs.")
    parser.add_argument("--images-dir", type=Path, help="Root directory containing city subfolders with GeoTIFFs.")
    parser.add_argument("--masks-dir", type=Path, help="Directory containing city GeoJSON mask files.")
    parser.add_argument("--cities", nargs="+", help="Optional list of city names to process from the images directory.")
    parser.add_argument("--annotation-crs", type=str, help="CRS of the GeoJSON (if not embedded).")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory where outputs are saved.")
    parser.add_argument(
        "--sam-checkpoint",
        type=Path,
        default=Path("sam2.1_b.pt"),
        help="Path to the SAM 2.1 checkpoint.",
    )
    parser.add_argument("--chip-buffer", type=int, default=128, help="Pixel buffer around each polygon chip.")
    parser.add_argument("--display-bands", type=int, nargs=3, help="Raster band indices for RGB display (1-based).")
    parser.add_argument("--save-mask-tif", action="store_true", help="Write GeoTIFF masks alongside GeoJSON.")
    parser.add_argument("--autosave-every", type=int, default=1, help="Autosave cadence in number of accepted buildings.")
    parser.add_argument(
        "--prompt-mode",
        choices=("points", "box", "mask"),
        default="points",
        help="Initial SAM prompt type: centroid point, bounding box, or full polygon mask.",
    )
    args = parser.parse_args()
    return args


def determine_entries(args: argparse.Namespace) -> List[Tuple[Path, Path]]:
    if args.batch_manifest:
        return load_manifest(args.batch_manifest)
    if args.image and args.annotations:
        return [(args.image, args.annotations)]
    if args.images_dir and args.masks_dir:
        return discover_city_entries(args.images_dir, args.masks_dir, args.cities)
    raise ValueError("Provide either --image/--annotations, --batch-manifest, or --images-dir/--masks-dir.")


def main() -> None:
    args = parse_args()
    entries = determine_entries(args)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app = SamAnnotatorApp(entries, args, device)
    app.run()


if __name__ == "__main__":
    main()
