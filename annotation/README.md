# SAM Building Refiner

Interactive tool for refining building footprints using a GeoTIFF image, Overture building polygons (GeoJSON), and the Segment Anything Model (SAM). It rasterizes the polygons to create an initial mask prior for SAM, then lets annotators iteratively fix results with positive/negative clicks and re-segmentation.

## What It Does

- Loads a GeoTIFF via rasterio and builds an 8‑bit RGB display tile (percentile stretch).
- Reads Overture GeoJSON, reprojects to the image CRS, filters to the image footprint, and creates per‑building masks.
- Presents buffered chips per building with the Overture polygon overlay; SAM only runs if you press “Re‑segment”.
- Simplified GUI flow: accept the prior mask, optionally refine with +/− clicks, skip if needed, and move to the next building/image.
- Saves outputs per image:
  - GeoJSON features in the image CRS with refined geometry and annotation metadata.
  - Optional single‑band mask GeoTIFFs aligned to the image (same transform/CRS).
  - Audit JSON of all clicks and parameters.

## Install

1) Python 3.10+ recommended. Create/activate an environment, then:

```
pip install -r requirements.txt
```

2) SAM checkpoint. Place `sam_b.pt` (or another Ultralytics SAM checkpoint) in the project directory or point to it with `--sam-checkpoint`.

GPU: If you have CUDA, install a CUDA build of PyTorch first (example for CUDA 12.1):

```
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Then install the rest via `pip install -r requirements.txt`.

## Quick Start (Single Image)

```
python SIMPL_Annotater_v2.py \
  --image /path/to/image.tif \
  --annotations /path/to/overture_buildings.geojson \
  --output-dir ./outputs \
  --sam-checkpoint ./sam_b.pt \
  --save-mask-tif
```

Optional flags:
- `--display-bands 4 3 2` to choose RGB bands (1‑based). Defaults to first 3 bands available.
- `--annotation-crs EPSG:4326` if the GeoJSON lacks CRS info and is known to be in WGS84.
- `--display-max-size 1400` to shrink the display window for very large rasters.
- `--autosave-every 1` autosave cadence in number of accepted buildings.
- `--percentile 2 98` for display stretch.
- `--chip-buffer 96` pixel margin around each building chip before feeding SAM.

## Folder Workflow (Multiple Cities)

Images directory layout:

```
/data/skysat_downloads/
├── Amsterdam/
│   ├── tile_01.tif
│   └── ...
├── Athens/
│   └── subtiles/.../*.tif
└── ...
```

Masks directory contains one GeoJSON per city (e.g. `/data/overture_masks/Amsterdam_buildings.geojson`).

Run:

```
python SIMPL_Annotater_v2.py \
  --images-dir /data/skysat_downloads \
  --masks-dir /data/overture_masks \
  --output-dir ./outputs \
  --sam-checkpoint ./sam_b.pt
```

- Use `--cities Amsterdam Athens` to limit the run to specific cities.
- The script matches each city subfolder to `*{city}*_buildings.geojson`. Missing cities/masks are reported and skipped.

## Batch Mode

Use a manifest file when annotating multiple images. Two formats are supported:

- JSON list:

```
[
  {"image": "/data/scene_a.tif", "annotations": "/data/overture_a.geojson"},
  {"image": "/data/scene_b.tif", "annotations": "/data/overture_b.geojson"}
]
```

- CSV (headers: `image,annotations`):

```
image,annotations
/data/scene_a.tif,/data/overture_a.geojson
/data/scene_b.tif,/data/overture_b.geojson
```

Run:

```
python SIMPL_Annotater_v2.py --batch-manifest manifest.json --output-dir ./outputs --sam-checkpoint ./sam_b.pt
```

## Controls

- Per image, the tool iterates buildings automatically and shows a buffered chip with the Overture mask overlay.
- Left‑click adds prompts (mode toggle on toolbar). Scroll/trackpad zooms; right/middle drag pans.
- Buttons / keys:
  - `a` Re‑segment (runs SAM only when you need it)
  - `s` Accept current mask and advance
  - `k` Skip building
  - `n` / `p` Next / Previous building
  - `z` Undo last prompt (Reset button clears all prompts)
  - `m` Toggle mask overlay to inspect imagery underneath
  - `Accept Nearby` button accepts all neighboring buildings currently shown
  - `Save & Next Image` writes outputs and advances to the next GeoTIFF

Notes:
- Default prompts (boundary positives + ring negatives and hole negatives) come from the Overture polygon; SAM is only invoked if you press `a`.
- Neighboring building outlines stay visible for context, making it easy to mass-accept consistent blocks.
- Accepted chips are stitched back into the full-resolution mask and exported with the original CRS.

## Outputs

Written to `--output-dir` per image:
- `<image_stem>_refined.geojson` — refined geometries in the image CRS with properties:
  - `building_id`, `source="overture"`, `refined_by="sam+human"`, `timestamp`, `mask_sum_pixels`, and arrays of positive/negative clicks.
- `<image_stem>_audit.json` — click history and metadata per accepted building.
- Optional: `<image_stem>_<building_id>_mask.tif` — single‑band `uint8` mask aligned to the source image.

## CRS / Reprojection

- The GeoTIFF’s CRS is used as the working CRS; GeoJSON polygons are reprojected into it on load.
- Exported GeoJSON and mask GeoTIFFs are in the GeoTIFF’s source CRS and align with the pixel grid.
- If your GeoJSON lacks a CRS, pass `--annotation-crs` (e.g., `EPSG:4326`).

## Troubleshooting

- `ModuleNotFoundError`: run `pip install -r requirements.txt` in your active environment.
- Very large images: increase `--display-max-size` for better context or reduce it to speed up drawing.
- Slow SAM inference: ensure your PyTorch install matches your GPU/CUDA. Otherwise it will run on CPU.
- No polygons displayed: make sure the Overture GeoJSON intersects the GeoTIFF footprint and IDs/properties are valid.

## File Layout

- `SIMPL_Annotater_v2.py` — main app and GUI.
- `requirements.txt` — pinned dependencies for Python installation.

## Citation

If you use this tool in research, please cite the Overture dataset and SAM accordingly.
