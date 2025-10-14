import os
import requests
import pandas as pd
from shapely.geometry import shape, box
from shapely.ops import unary_union
import time

# --- Configuration ---
API_KEY = os.getenv("PL_API_KEY")
if not API_KEY:
    raise ValueError("No PL_API_KEY found in environment variables")

# Mode selection: 
# - 'search': Search for scenes based on skysat_coverage_weilin.csv
# - 'download_csv': Download scenes directly from random_5_scenes_per_city.csv
RUN_MODE = 'download_csv'  # Options: 'search', 'download_csv'

SEARCH_URL = "https://api.planet.com/data/v1/quick-search"
ITEM_TYPE = "SkySatScene"
ASSET_TYPE = "ortho_visual"  
DATE_RANGE = {"gte": "2023-01-01T00:00:00Z", "lte": "2025-12-31T00:00:00Z"}
MAX_CLOUD_COVER = 0.05
DEG_HALF_SIDE = 0.045  # 10km box

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

def search_all_scenes(lat, lon, api_key):
    """Search for all scenes for a given lat/lon."""
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

def download_scene(scene_id, city, api_key):
    """Activate and download a single scene."""
    item_url = f"https://api.planet.com/data/v1/item-types/{ITEM_TYPE}/items/{scene_id}/assets"
    
    # 1. Get asset list
    time.sleep(0.1) # Respect "other" endpoint rate limit (10/sec)
    r_assets = make_request(item_url, auth=(api_key, ""))
    if not r_assets:
        print(f"Failed to get assets for scene {scene_id}")
        return
    assets = r_assets.json()
    
    if ASSET_TYPE not in assets:
        print(f"Asset type '{ASSET_TYPE}' not available for scene {scene_id}")
        return

    asset = assets[ASSET_TYPE]
    
    # 2. Activate asset if not already active
    if asset['status'] != 'active':
        activation_url = asset["_links"]["activate"]
        time.sleep(0.5) # Respect activation rate limit (2/sec)
        r_activate = make_request(activation_url, auth=(api_key, ""))
        if not r_activate:
            print(f"Failed to send activation request for {scene_id}")
            return
        
        # 3. Poll for activation status
        print(f"Activating {scene_id}...")
        start_time = time.time()
        while True:
            time.sleep(0.5) # Poll activation status
            r_status = make_request(asset["_links"]["_self"], auth=(api_key, ""))
            if not r_status:
                print(f"Failed to get activation status for {scene_id}")
                return
            
            status = r_status.json()["status"]
            if status == 'active':
                elapsed_time = time.time() - start_time
                print(f"Activation complete for {scene_id} in {elapsed_time:.2f} seconds.")
                break
            if status == 'failed':
                print(f"Activation failed for {scene_id}.")
                return
            # Continue polling if 'activating' or 'inactive'
            time.sleep(10) # Wait longer between polls
    else:
        print(f"Asset {scene_id} is already active.")

    # 4. Download asset
    # Re-fetch asset data to get the download location
    r_status = make_request(asset["_links"]["_self"], auth=(api_key, ""))
    if not r_status or r_status.json().get('status') != 'active':
        print(f"Could not retrieve active asset details for {scene_id}")
        return
        
    download_url = r_status.json()["location"]
    
    download_dir = f"/scratch/skysat_downloads/{city}"
    os.makedirs(download_dir, exist_ok=True)
    
    filename = f"{scene_id}_{ASSET_TYPE}.tif"
    filepath = os.path.join(download_dir, filename)
    
    print(f"Downloading {filename} to {filepath}...")
    time.sleep(0.2) # Respect download rate limit (5/sec)
    with make_request(download_url, stream=True, auth=(api_key, "")) as r_download:
        if r_download:
            with open(filepath, 'wb') as f:
                for chunk in r_download.iter_content(chunk_size=8192):
                    f.write(chunk)
            print(f"Download complete for {scene_id}.")
        else:
            print(f"Failed to download {scene_id}.")

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
    """Search for scenes by city and then download them."""
    print("--- Running in SEARCH and DOWNLOAD mode ---")
    df = pd.read_csv("skysat_coverage_weilin.csv")
    for _, row in df.iterrows():
        city = row['City']
        print(f"\n--- Processing {city} ---")
        features, _ = search_all_scenes(row['Latitude'], row['Longitude'], API_KEY)
        
        if not features:
            print(f"{city}: 0 scenes found.")
        else:
            print(f"{city}: Found {len(features)} scenes. Starting download...")
            for feature in features:
                download_scene(feature['id'], city, API_KEY)
                time.sleep(1) # Pause between downloading different scenes
        time.sleep(1)

def run_download_from_csv():
    """Download scenes directly from a CSV file."""
    print("--- Running in DOWNLOAD FROM CSV mode ---")
    df = pd.read_csv("random_5_scenes_per_city_saad_bottom_half.csv")
    for _, row in df.iterrows():
        city = row['City']
        scene_id = row['Scene_ID']
        print(f"\n--- Processing scene {scene_id} for {city} ---")
        download_scene(scene_id, city, API_KEY)
        time.sleep(1) # Pause between downloading different scenes

if __name__ == "__main__":
    if RUN_MODE == 'search':
        run_search_and_download()
    elif RUN_MODE == 'download_csv':
        run_download_from_csv()
    else:
        print(f"Invalid RUN_MODE: '{RUN_MODE}'. Choose 'search' or 'download_csv'.")