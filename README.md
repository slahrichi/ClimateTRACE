# Quick Start: SIMPL Annotator (SAM Building Refiner)

## 0. Upcoming changes

- [ ] Dropping Ultralytics and using Meta's repo to call SAM 
- [ ] Fixing SAM Mask prompt bug  
- [ ] Allowing user to choose between Vanilla SAM or finetuned SAM 
- [ ] Batch processing buildings in the same chip by prompting SAM in parallel
- [ ] Letting SAM participate in segmenting newly-created buildings


## 1. Install the Annotator

```bash
git clone https://github.com/slahrichi/ClimateTRACE
cd ClimateTRACE
conda create -n annotator python=3.10 -y
conda activate annotator
pip install -r requirements.txt
```
Or use uv / venv (see full README for more)

---
**GPU users**: Install a CUDA-enabled PyTorch build before running
pip install -r requirements.txt.


## 2. Download SkySat Imagery + Overture Masks

Download the dataset from OneDrive: https://mailmissouri-my.sharepoint.com/:f:/g/personal/slxg3_umsystem_edu/EpUDvN79-mxChwI8fZENeLMBfm2Nja6AHmUrIsTDqDJ3IQ

The folder includes:

* `Your_Name` -- folder for each annotator's data 

* `Images` — .tif SkySat chips to annotate

* `Masks` — .geoJSON Overture building polygons 


Download them anywhere on your system, e.g.:

```bash
/data/Images
/data/Masks
```

## 3. Run the Annotator (City Folder Mode)

To annotate the cities you are responsible for:

```bash
python SIMPL_Annotater_v2.py \
  --images-dir /data/Images \
  --masks-dir /data/Masks \
  --output-dir outputs/ \
  --sam-checkpoint sam2.1_b.pt \
  --annotator-id YourName \
```

## 4. Annotate 
The script iterates through all the buildings in the image, (left-side panel has all the building ids). On each building, we overlay the prior mask from Overture (blue outline, red mask). You can toggle the mask view on/off by pressing `m`.  We also show the nearby building using a purple outline. 

**You need to decide, for each building mask, whether it's correct (accept `s`), needs to be deleted (delete `d`), or needs to be refined.**

To refine a mask, you can either use the brush (`b`) and eraser (`e`) and manually edit the mask, or use SAM (re-segment `a`). If you don't provide any prompt, SAM will be fed a center point or a box around the mask (depending on chosen prompt style). Otherwise, you can add positive/negative points then prompt SAM and get a new mask. Once satisfied with the mask, you can accept it and move to the next building. 

**If a building is missing, i.e., exists in the image but does not have a mask, you can use the create mode (`c`) and manually create a new building, then save it (`v`)**

## 5. Upload your results

Once done, upload your results in the same OneDrive shared folder, under ```YourName/output```. This should include the refined.geojson with the final polygons, the deleted.geojson with the polygons marked for deletion, the new.geojson with the newly-created buildings, and the audit.json log file.

## 6. Tips
- If you want to pause and resume later, simply call the script usign the same call (same output folder, and your previous annotations will be loaded)

- If all the building masks in the chip look correct (the current one in blue, and all the ones next to it in purple), you can simply click Accept Nearby `h` and all will be accepted. Only use if all buildings look correct. 



-----------------------------------------

# Full README

Interactive annotator for refining building footprints. It loads a GeoTIFF image + GeoJSON rough mask (from Overture), clips each polygon into a chip, and lets you adjust masks with Segment Anything 2.1 (points/boxes/masks) plus manual brushes, deletions, and new-building creation. Every edit is tracked with annotator IDs and timing metrics.

---

## Features

- **Chip-by-chip SAM prompts** – centroid point, bounding box, or seed mask, plus interactive positive/negative clicks.
- **Edit tools** – brush/eraser to touch up masks, connected-component delete, and a creation workflow (line or brush) for missing buildings.
- **Autosave + logging** – every accept/delete/create writes to GeoJSON immediately; additional files summarize deleted and newly created features plus a full audit log.
- **Keyboard-first UI** – all buttons display their shortcut (e.g., `a` re-segments, `s` accepts, `w` saves & advances).
- **Annotator + timing metadata** – prompts for an ID (or use `--annotator-id`) and records per-building/image/session durations for QA.

---

## Installation

1. Clone or copy the repository locally.
2. Use Python 3.10+ and install dependencies.

### Conda

```bash
conda create -n annotator python=3.10 -y
conda activate annotator
pip install -r requirements.txt
```

### uv

```bash
uv venv .venv --python 3.10
source .venv/bin/activate
uv pip install -r requirements.txt
```

### System pip / venv

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

> **SAM weights:** Place a SAM 2.1 checkpoint (e.g., `sam2.1_b.pt`) in the repo or pass the path with `--sam-checkpoint`.

GPU users should install a CUDA-enabled PyTorch build before `pip install -r requirements.txt`.

---

## How to Run

`SIMPL_Annotater_v2.py` accepts a single image pair or city directories.

### Single Image

```bash
python SIMPL_Annotater_v2.py \
  --image /path/to/city.tif \
  --annotations /path/to/city_buildings.geojson \
  --output-dir outputs/city \
  --sam-checkpoint sam2.1_b.pt \
  --annotator-id Jordan
```


### City Folder Discovery

```bash
python SIMPL_Annotater_v2.py \
  --images-dir /data/skysat_downloads \
  --masks-dir overture_masks \
  --cities Amsterdam Athens \
  --output-dir outputs/cities \
  --sam-checkpoint sam2.1_b.pt
```

### Common Flags

- `--display-bands 4 3 2` – choose RGB bands (1-based).
- `--chip-buffer 128` – pixel padding around each polygon chip.
- `--prompt-mode {points|box|mask}` – default SAM prompt type (mask is still broken and will be fixed in the next update)
- `--save-mask-tif` – write per-building GeoTIFF masks. (slow, not recommended unless we need to inspect individual building masks)
- `--autosave-every N` – autosave cadence (defaults to 1; autosave also triggers after every accept/delete/create).

When the UI starts, it prompts for an annotator ID unless you passed `--annotator-id`. The main window appears and begins loading the first image.

### Resuming a Session

Run the same command again (same `--output-dir`). The tool reads any existing `<stem>_refined.geojson`, `<stem>_deleted.geojson`, and `<stem>_new.geojson`, marks previously accepted/deleted buildings, reloads newly created footprints, and jumps to the first unfinished polygon.

---

## Annotating Workflow

1. **Select a building** from the list (left). The chip appears with overlays.
2. **Inspect default mask** (derived from the polygon). Use the mouse wheel or `+/-` to zoom; middle/right drag to pan.
3. **Prompt SAM** (`a` / “Re-segment (a)”) using positive/negative clicks. Toggle between point modes on the toolbar. Segmenting 
4. **Accept (`s`) or skip (`k`)**. Accepting writes the refined mask, audit entry, and metadata immediately.
5. **Save & next image** (`w`) when you’ve finished all buildings in a GeoTIFF.

### Brush / Eraser

- `Brush Edit (b)` or `Eraser Edit (e)` to enter drawing mode; drag with the mouse, adjust radius via the spinbox.
- `Done Editing` exits.

### Delete Mode

- `Delete (d)` → click inside the building to flood-fill that component away. Entire deletions log to `<stem>_deleted.geojson`.

### Create Mode

- `Create (c)` → choose Line (click vertices) or Brush (paint). `Save New (v)` commits a new `user_building_XXXX` feature; `Cancel New (u/Esc)` aborts.

### Accept Nearby

- `Accept Nearby (h)` automatically accepts any overlapping polygons currently in view.

---

## Keyboard Shortcuts

| Action | Shortcut | Button |
|--------|----------|--------|
| Re-segment | `a` | Re-segment (a) |
| Accept mask | `s` | Accept (s) |
| Skip building | `k` | Skip (k) |
| Next / Previous | `n` / `p` | – |
| Undo / Reset | `z` / `r` | Undo (z) / Reset clicks (r) |
| Save & Next image | `w` | Save & Next (w) |
| Toggle mask | `m` | Toggle Mask (m) |
| Delete mode | `d` | Delete (d) |
| Create mode | `c` | Create (c) |
| Save New / Cancel New | `v` / `u` / `Esc` | Save New (v), Cancel New (u / Esc) |
| Brush / Eraser edit | `b` / `e` | Brush Edit (b) / Eraser Edit (e) |
| Accept Nearby | `h` | Accept Nearby (h) |
| Save & exit session | `q` | Exit (q) |

Buttons display their shortcut in the label for easy reference.

---

## Outputs

Written per image to `--output-dir`:

- `<stem>_refined.geojson` – final building geometries in the image CRS. Each feature includes:
  - `building_id`, `source` (preserved from the polygon), `refined_by`, `timestamp`, `mask_sum_pixels`.
  - Annotator metadata and timing metrics: `annotator_id`, `building_elapsed_sec`, `image_elapsed_sec`, `session_elapsed_sec`.
  - Lists of positive/negative clicks for traceability.
- `<stem>_deleted.geojson` – features removed during annotation (with metadata).
- `<stem>_new.geojson` – new polygons you created.
- `<stem>_audit.json` – chronological log (annotator ID, clicks, prompts, runtimes, actions).
- Optional `<stem>_<building_id>_mask.tif` if `--save-mask-tif` is set.

Autosave runs automatically after every accept/delete/create, when switching images, and on exit.


## Acknowledgments
Thanks to Taylor McKechnie for providing the foundation code we built the current script on.


## Happy annotating! Contributions/bug reports are welcome—open an issue or submit a PR if you extend the workflow.

