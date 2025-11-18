import os
import requests
import pandas as pd
import rasterio
from shapely.geometry import shape, box, Polygon
from shapely.ops import unary_union
import time
from datetime import datetime
from tqdm import tqdm
from pyproj import Proj, transform, CRS

# --- Configuration ---
API_KEY = os.getenv("PL_API_KEY")
if not API_KEY:
    raise ValueError("No PL_API_KEY found in environment variables")

# Mode selection: 
# - 'search': Search for SkySatCollect items based on skysat_coverage_weilin.csv
# - 'download_csv': Download SkySatCollect items directly from a CSV
RUN_MODE = 'search'  # Options: 'search', 'download_csv'

SEARCH_URL = "https://api.planet.com/data/v1/quick-search"
ITEM_TYPE = "SkySatCollect"
ASSET_TYPE = "ortho_visual"  # Examples: 'ortho_visual', 'ortho_analytic', 'ortho_analytic_udm2'
DATE_RANGE = {"gte": "2023-10-01T00:00:00Z", "lte": "2023-12-31T00:00:00Z"}
MAX_CLOUD_COVER = 0.01
DEG_HALF_SIDE = 0.045  # 10km box
LOG_FILE_NAME = "skysat_collect_download_timings.txt"
DOWNLOAD_LOGS = []
DOWNLOAD_AREA_LIMIT_SQKM = 2750
DOWNLOADS_PER_CITY_LIMIT = 3


def get_scene_area_sqkm(tif_path):
    """Calculates the area of a raster in square kilometers."""
    try:
        with rasterio.open(tif_path) as src:
            bounds = src.bounds
            src_crs = src.crs
            if not src_crs:
                src_crs = CRS.from_epsg(4326)  # Assume WGS84 if no CRS is found

            geom_poly = Polygon([
                (bounds.left, bounds.bottom),
                (bounds.left, bounds.top),
                (bounds.right, bounds.top),
                (bounds.right, bounds.bottom)
            ])

            src_proj = Proj(src_crs)
            dst_proj = Proj('+proj=cea')  # Cylindrical Equal Area

            projected_poly_points = [transform(src_proj, dst_proj, x, y) for x, y in geom_poly.exterior.coords]
            projected_poly = Polygon(projected_poly_points)

            return projected_poly.area / 1_000_000
    except Exception as e:
        print(f"Could not calculate area for {os.path.basename(tif_path)}: {e}")
        return 0

def calculate_initial_area(download_dir):
    """Calculates the total area of all existing .tif files in the download directory."""
    total_area = 0
    print(f"Calculating initial area from existing files in {download_dir}...")
    if not os.path.isdir(download_dir):
        return 0
        
    for city_folder in os.listdir(download_dir):
        city_path = os.path.join(download_dir, city_folder)
        if os.path.isdir(city_path):
            for filename in os.listdir(city_path):
                if filename.lower().endswith('.tif'):
                    filepath = os.path.join(city_path, filename)
                    total_area += get_scene_area_sqkm(filepath)
    print(f"Initial area is {total_area:.2f} sq. km.")
    return total_area


def record_download_log(scene_id, city, activation_seconds, download_seconds, total_seconds, status, note="", filepath=""):
    """Add a timing record for later persistence."""
    DOWNLOAD_LOGS.append(
        {
            "timestamp": datetime.utcnow().isoformat(),
            "scene_id": scene_id,
            "city": city,
            "activation_seconds": activation_seconds,
            "download_seconds": download_seconds,
            "total_seconds": total_seconds,
            "status": status,
            "note": note,
            "filepath": filepath,
        }
    )


def write_download_log():
    """Persist download timing details to a text file."""
    log_timestamp = datetime.utcnow().isoformat()
    try:
        with open(LOG_FILE_NAME, "w", encoding="utf-8") as log_file:
            log_file.write(f"# SkySat download timing log generated at {log_timestamp} UTC\n")
            log_file.write("timestamp,scene_id,city,activation_seconds,download_seconds,total_seconds,status,filepath,note\n")
            if not DOWNLOAD_LOGS:
                log_file.write("# No downloads processed during this run.\n")
            else:
                for entry in DOWNLOAD_LOGS:
                    activation = "" if entry["activation_seconds"] is None else f"{entry['activation_seconds']:.2f}"
                    download = "" if entry["download_seconds"] is None else f"{entry['download_seconds']:.2f}"
                    total = "" if entry["total_seconds"] is None else f"{entry['total_seconds']:.2f}"
                    filepath = entry["filepath"] or ""
                    note = (entry["note"] or "").replace("\n", " ").strip()
                    line = (
                        f"{entry['timestamp']},"
                        f"{entry['scene_id']},"
                        f"{entry['city']},"
                        f"{activation},"
                        f"{download},"
                        f"{total},"
                        f"{entry['status']},"
                        f"{filepath},"
                        f"{note}\n"
                    )
                    log_file.write(line)
        if DOWNLOAD_LOGS:
            print(f"Download timing details saved to {LOG_FILE_NAME}.")
        else:
            print(f"No downloads recorded; created empty log file {LOG_FILE_NAME}.")
    except Exception as exc:
        print(f"Failed to write download timing log: {exc}")




# --- Rate Limiting and Retries ---
def make_request(url, method='get', auth=None, json=None, stream=False, max_retries=2):
    """Makes an HTTP request with rate limiting and retry logic."""
    retries = 0
    while retries <= max_retries:
        try:
            if method == 'post':
                r = requests.post(url, auth=auth, json=json, timeout=60)
            else:
                r = requests.get(url, auth=auth, stream=stream, timeout=60)
            
            if r.status_code == 429: # Rate limit exceeded
                print("Rate limit hit. Waiting and retrying...")
                time.sleep(5) 
                retries += 1
                continue

            r.raise_for_status()
            return r

        except requests.exceptions.RequestException as e:
            print(f"Request failed: {e}. Retrying ({retries+1}/{max_retries})...")
            retries += 1
            time.sleep(2) # Wait before retrying
    
    print(f"Request failed after {max_retries} retries.")
    return None

# --- Core Functions ---
def make_bbox(lat, lon, half_side=DEG_HALF_SIDE):
    """Create target 10x10km box"""
    geom = box(lon - half_side, lat - half_side, lon + half_side, lat + half_side)
    return geom, geom.__geo_interface__

def search_all_collects(lat, lon, api_key):
    """Search for all SkySatCollect items for a given lat/lon."""
    target_box, aoi = make_bbox(lat, lon)
    
    query = {
        "item_types": [ITEM_TYPE],
        "filter": {
            "type": "AndFilter",
            "config": [
                {"type": "GeometryFilter", "field_name": "geometry", "config": aoi},
                {"type": "DateRangeFilter", "field_name": "acquired", "config": DATE_RANGE},
                {"type": "RangeFilter", "field_name": "cloud_cover", "config": {"lte": MAX_CLOUD_COVER}}
            ]
        }
    }
    
    all_features = []
    
    # Initial search request
    time.sleep(0.2) # Respect search rate limit (5/sec)
    r = make_request(SEARCH_URL, method='post', auth=(api_key, ""), json=query)
    if not r:
        return [], target_box
        
    response_data = r.json()
    features = response_data.get("features", [])
    all_features.extend(features)
    
    next_url = response_data.get("_links", {}).get("_next")
    page = 1
    
    # Paginate through results
    while next_url and page < 50:
        page += 1
        time.sleep(0.2) # Respect search rate limit
        r = make_request(next_url, auth=(api_key, ""))
        if r:
            response_data = r.json()
            features = response_data.get("features", [])
            all_features.extend(features)
            next_url = response_data.get("_links", {}).get("_next")
        else:
            break
    
    return all_features, target_box

def download_collect(collect_id, city, api_key):
    """Activate and download a single SkySatCollect asset, and return its area."""
    overall_start = time.time()
    activation_elapsed = None
    download_elapsed = None
    filepath = ""
    area_sqkm = 0

    def log_and_return(status, note=""):
        total_elapsed = time.time() - overall_start
        record_download_log(
            collect_id,
            city,
            activation_elapsed,
            download_elapsed,
            total_elapsed,
            status,
            note,
            filepath,
        )
        return area_sqkm

    try:
        item_url = f"https://api.planet.com/data/v1/item-types/{ITEM_TYPE}/items/{collect_id}/assets"
        
        # 1. Get asset list
        time.sleep(0.1) # Respect "other" endpoint rate limit (10/sec)
        r_assets = make_request(item_url, auth=(api_key, ""))
        if not r_assets:
            print(f"Failed to get assets for collect {collect_id}")
            return log_and_return("failed", "Failed to get assets")
        assets = r_assets.json()
        
        if ASSET_TYPE not in assets:
            print(f"Asset type '{ASSET_TYPE}' not available for collect {collect_id}")
            return log_and_return("failed", f"Asset type '{ASSET_TYPE}' unavailable")

        asset = assets[ASSET_TYPE]
        
        # 2. Activate asset if not already active
        if asset['status'] != 'active':
            activation_url = asset["_links"]["activate"]
            time.sleep(0.5) # Respect activation rate limit (2/sec)
            # Activation requires POST per Planet Data API
            r_activate = make_request(activation_url, method='post', auth=(api_key, ""))
            if not r_activate:
                print(f"Failed to send activation request for {collect_id}")
                return log_and_return("failed", "Activation request failed")
            
            # 3. Poll for activation status
            print(f"Activating {collect_id}...")
            activation_start = time.time()
            while True:
                time.sleep(0.5) # Poll activation status
                r_status = make_request(asset["_links"]["_self"], auth=(api_key, ""))
                if not r_status:
                    print(f"Failed to get activation status for {collect_id}")
                    return log_and_return("failed", "Activation status unavailable")
                
                status = r_status.json()["status"]
                if status == 'active':
                    activation_elapsed = time.time() - activation_start
                    print(f"Activation complete for {collect_id} in {activation_elapsed:.2f} seconds.")
                    break
                if status == 'failed':
                    print(f"Activation failed for {collect_id}.")
                    return log_and_return("failed", "Activation failed")
                # Continue polling if 'activating' or 'inactive'
                time.sleep(10) # Wait longer between polls
        else:
            print(f"Asset {collect_id} is already active.")
            activation_elapsed = 0.0

        # 4. Download asset
        # Re-fetch asset data to get the download location
        r_status = make_request(asset["_links"]["_self"], auth=(api_key, ""))
        if not r_status or r_status.json().get('status') != 'active':
            print(f"Could not retrieve active asset details for {collect_id}")
            return log_and_return("failed", "Active asset details unavailable")
            
        download_url = r_status.json()["location"]
        
        download_dir = f"/Users/slxg3/Downloads/PhD/RA/ClimateTrace/skysat_downloads/{city}"
        os.makedirs(download_dir, exist_ok=True)
        
        filename = f"{collect_id}_{ASSET_TYPE}.tif"
        filepath = os.path.join(download_dir, filename)

        if os.path.exists(filepath):
            print(f"File {filepath} already exists. Skipping download.")
            return log_and_return("skipped", "File already exists")
        
        print(f"Downloading {filename} to {filepath}...")
        time.sleep(0.2) # Respect download rate limit (5/sec)
        download_start = time.time()
        r_download = make_request(download_url, stream=True, auth=(api_key, ""))
        if not r_download:
            print(f"Failed to download {collect_id}.")
            return log_and_return("failed", "Download request failed")
        
        try:
            with open(filepath, 'wb') as f:
                for chunk in r_download.iter_content(chunk_size=8192):
                    f.write(chunk)
            download_elapsed = time.time() - download_start
            print(f"Download complete for {collect_id}.")
            area_sqkm = get_scene_area_sqkm(filepath)
            log_and_return("success", f"Area: {area_sqkm:.2f} sqkm")
        except Exception as exc:
            print(f"Failed to write file for {collect_id}: {exc}")
            log_and_return("failed", f"File write failed: {exc}")
        finally:
            if r_download:
                r_download.close()
        
        return area_sqkm

    except Exception as exc:
        print(f"Unexpected error while processing {collect_id}: {exc}")
        return log_and_return("error", f"Unexpected error: {exc}")

def calculate_coverage(features, target_box):
    """Calculate coverage percentage"""
    if not features:
        return 0, 0
    
    try:
        scene_geometries = [shape(f['geometry']) for f in features]
        combined = unary_union(scene_geometries)
        intersection = combined.intersection(target_box)
        coverage_pct = (intersection.area / target_box.area) * 100
        return coverage_pct, len(features)
    except Exception as e:
        print(f"Error in calculate_coverage: {e}")
        return 0, len(features)

# --- Main Execution Logic ---
def run_search_and_download():
    """Search for SkySatCollect items by city and then download them."""
    print("--- Running in SEARCH and DOWNLOAD mode ---")
    
    download_dir = f"/Users/slxg3/Downloads/PhD/RA/ClimateTrace/skysat_downloads"
    total_area_sqkm = calculate_initial_area(download_dir)

    csv_name = "skysat_coverage_weilin.csv"
    df = pd.read_csv(csv_name)
    
    if df.empty:
        print(f"No rows found in {csv_name}.")
        return

    for _, row in tqdm(df.iterrows(), total=df.shape[0], desc=f"Processing {csv_name}"):
        if total_area_sqkm >= DOWNLOAD_AREA_LIMIT_SQKM:
            print(f"\nTotal area limit of {DOWNLOAD_AREA_LIMIT_SQKM} sq. km reached. Halting downloads.")
            break

        city = row['City']
        print(f"\n--- Processing {city} ---")
        features, _ = search_all_collects(row['Latitude'], row['Longitude'], API_KEY)
        
        if not features:
            print(f"{city}: 0 collects found.")
        else:
            print(f"{city}: Found {len(features)} collects. Sorting by cloud cover and applying limit of {DOWNLOADS_PER_CITY_LIMIT}.")
            # Sort features by cloud cover (ascending)
            features.sort(key=lambda f: f['properties'].get('cloud_cover', 1.0))
            features_to_download = features[:DOWNLOADS_PER_CITY_LIMIT]
            
            for feature in tqdm(features_to_download, desc=f"Downloading collects for {city}"):
                if total_area_sqkm >= DOWNLOAD_AREA_LIMIT_SQKM:
                    print(f"Total area limit reached during download for {city}. Halting.")
                    break
                
                area_downloaded = download_collect(feature['id'], city, API_KEY)
                total_area_sqkm += area_downloaded
                remaining_area = DOWNLOAD_AREA_LIMIT_SQKM - total_area_sqkm
                print(f"Running total area: {total_area_sqkm:.2f} / {DOWNLOAD_AREA_LIMIT_SQKM} sq. km. Remaining: {remaining_area:.2f} sq. km.")
                time.sleep(1)
        time.sleep(1)

def run_download_from_csv():
    """Download SkySatCollect items directly from a CSV file.

    Supports either a 'Collect_ID' column (preferred) or a legacy 'Scene_ID' column.
    """
    print("--- Running in DOWNLOAD FROM CSV mode ---")
    
    download_dir = f"/Users/slxg3/Downloads/PhD/RA/ClimateTrace/skysat_downloads"
    total_area_sqkm = calculate_initial_area(download_dir)

    csv_name = "/Users/slxg3/Downloads/PhD/RA/ClimateTrace/ClimateTRACE/random_5_scenes_per_city_doaa_top.csv"
    df = pd.read_csv(csv_name)

    if df.empty:
        print(f"No rows found in {csv_name}.")
        return

    # Group by city and limit scenes per city
    df_limited = df.groupby('City').head(DOWNLOADS_PER_CITY_LIMIT).reset_index()

    # Determine which column to use for IDs
    id_col = None
    if 'Collect_ID' in df_limited.columns:
        id_col = 'Collect_ID'
    elif 'Scene_ID' in df_limited.columns:
        id_col = 'Scene_ID'  # legacy support
    else:
        print("CSV must contain either 'Collect_ID' or 'Scene_ID' column.")
        return

    for _, row in tqdm(df_limited.iterrows(), total=df_limited.shape[0], desc=f"Processing {csv_name}"):
        if total_area_sqkm >= DOWNLOAD_AREA_LIMIT_SQKM:
            print(f"\nTotal area limit of {DOWNLOAD_AREA_LIMIT_SQKM} sq. km reached. Halting downloads.")
            break

        city = row['City']
        collect_id = row[id_col]
        print(f"\n--- Processing collect {collect_id} for {city} ---")
        
        area_downloaded = download_collect(collect_id, city, API_KEY)
        total_area_sqkm += area_downloaded
        remaining_area = DOWNLOAD_AREA_LIMIT_SQKM - total_area_sqkm
        print(f"Running total area: {total_area_sqkm:.2f} / {DOWNLOAD_AREA_LIMIT_SQKM} sq. km. Remaining: {remaining_area:.2f} sq. km.")

        time.sleep(1) # Pause between downloading different scenes

if __name__ == "__main__":
    try:
        if RUN_MODE == 'search':
            run_search_and_download()
        elif RUN_MODE == 'download_csv':
            run_download_from_csv()
        else:
            print(f"Invalid RUN_MODE: '{RUN_MODE}'. Choose 'search' or 'download_csv'.")
    finally:
        write_download_log()
