import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import sys
import numpy as np
import os
from shapely.geometry import Polygon
import rasterio
from rasterio.warp import transform_bounds
import requests
import time
import json
import zipfile
import io

# Area of interest coordinates
aoi_coords = [
    [78.58013561221814,51.28376604439698], [78.20660045596814,50.994253517722704],
    [78.79986217471814,50.3537270276519], [78.40435436221814,50.04429404493862],
    [77.65728404971814,50.25548861230339], [76.86626842471814,49.44798023538978],
    [77.63531139346814,48.68500731531969], [78.62408092471814,47.93974579613136],
    [80.38189342471814,48.1454115464798], [81.87603404971814,49.57637905477649],
    [81.39263561221814,50.17112270356419], [81.59038951846814,50.63329091010904],
    [80.77740123721814,51.15991154397452], [80.11822154971814,50.66115634184031],
    [78.58013561221814,51.28376604439698]
]

REFERENCE_POINT = [80.23, 50.42]
start_date = '2023-04-01'
end_date = '2023-09-30'
SCALE = 200  # Processing scale in meters
CSV_FILE = "processed_data_2023_v2.csv"
TARGET_CELLS = 100  # Desired number of grid cells
TEMP_DIR = "temp_data"  # Directory for temporary files

# USGS Earth Explorer credentials - you'll need to register at https://earthexplorer.usgs.gov/
USGS_USERNAME = "your_username"
USGS_PASSWORD = "your_password"

def log(message):
    print(message)
    sys.stdout.flush()

def calculate_grid_size(aoi_coords, target_cells):
    min_lon = min(coord[0] for coord in aoi_coords)
    max_lon = max(coord[0] for coord in aoi_coords)
    min_lat = min(coord[1] for coord in aoi_coords)
    max_lat = max(coord[1] for coord in aoi_coords)

    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    cell_area = (lon_range * lat_range) / target_cells
    grid_size = cell_area ** 0.5
    return grid_size

def generate_grid(aoi_coords, grid_size):
    min_lon = min(coord[0] for coord in aoi_coords)
    max_lon = max(coord[0] for coord in aoi_coords)
    min_lat = min(coord[1] for coord in aoi_coords)
    max_lat = max(coord[1] for coord in aoi_coords)

    grid_cells = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            grid_cells.append(Polygon([
                (lon, lat),
                (lon + grid_size, lat),
                (lon + grid_size, lat + grid_size),
                (lon, lat + grid_size)
            ]))
            lon += grid_size
        lat += grid_size

    return grid_cells

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def polygon_to_wkt(polygon):
    """Convert a Shapely polygon to WKT format"""
    return polygon.wkt

def search_sentinel_images_usgs(date, aoi_polygon):
    """Search for Sentinel-2 images using USGS Earth Explorer"""
    # Convert polygon to WKT format
    footprint = polygon_to_wkt(aoi_polygon)
    
    # Format date for query
    date_obj = datetime.strptime(date, '%Y-%m-%d')
    next_date = (date_obj + timedelta(days=1)).strftime('%Y-%m-%d')
    
    # USGS Earth Explorer API endpoint
    url = "https://earthexplorer.usgs.gov/inventory/json"
    
    # Parameters for the request
    params = {
        "jsonRequest": json.dumps({
            "apiKey": "your_api_key",  # You'll need to get an API key from USGS
            "datasetName": "SENTINEL_2A",
            "spatialFilter": {
                "filterType": "mbr",
                "mbr": {
                    "west": min(coord[0] for coord in aoi_coords),
                    "east": max(coord[0] for coord in aoi_coords),
                    "north": max(coord[1] for coord in aoi_coords),
                    "south": min(coord[1] for coord in aoi_coords)
                }
            },
            "temporalFilter": {
                "startDate": date,
                "endDate": next_date
            },
            "maxResults": 10,
            "sortOrder": "ASC",
            "sortField": "acquisitionDate"
        })
    }
    
    # Make the request
    try:
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            log(f"Error searching for images: {response.status_code}")
            return None
        
        # Parse the response
        data = response.json()
        
        if 'data' not in data or 'results' not in data['data'] or not data['data']['results']:
            log(f"No images found for {date}")
            return None
        
        # Get the least cloudy image
        results = data['data']['results']
        best_result = None
        min_cloud = float('inf')
        
        for result in results:
            cloud_cover = float(result.get('cloudCover', 100))
            if cloud_cover < min_cloud:
                min_cloud = cloud_cover
                best_result = result
        
        if best_result:
            # Get the scene ID and download URL
            scene_id = best_result.get('entityId')
            download_url = best_result.get('downloadUrl')
            
            return {
                'id': scene_id,
                'download_url': download_url,
                'cloud_cover': min_cloud,
                'date': date
            }
        
        return None
    
    except Exception as e:
        log(f"Error searching for images: {e}")
        return None

def download_sentinel_image_usgs(image_info):
    """Download a Sentinel-2 image from USGS Earth Explorer"""
    if not image_info or 'download_url' not in image_info:
        return None
    
    # Create temporary directory if it doesn't exist
    os.makedirs(TEMP_DIR, exist_ok=True)
    
    # Download the image
    log(f"Downloading image {image_info['id']}...")
    try:
        # First, authenticate with USGS
        auth_url = "https://earthexplorer.usgs.gov/inventory/json"
        auth_params = {
            "jsonRequest": json.dumps({
                "username": USGS_USERNAME,
                "password": USGS_PASSWORD,
                "catalog": "EE"
            })
        }
        
        auth_response = requests.post(auth_url, data=auth_params)
        auth_data = auth_response.json()
        
        if 'error' in auth_data:
            log(f"Authentication error: {auth_data['error']}")
            return None
        
        # Now download the image
        response = requests.get(image_info['download_url'], stream=True)
        
        if response.status_code != 200:
            log(f"Error downloading image: {response.status_code}")
            return None
        
        # Save the zip file
        zip_path = os.path.join(TEMP_DIR, f"{image_info['id']}.zip")
        with open(zip_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        # Extract the zip file
        extract_path = os.path.join(TEMP_DIR, image_info['id'])
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(extract_path)
        
        # Find the SWIR and NIR band files
        swir_path = None
        nir_path = None
        
        for root, dirs, files in os.walk(extract_path):
            for file in files:
                if file.endswith('_B11.jp2'):  # SWIR band
                    swir_path = os.path.join(root, file)
                elif file.endswith('_B08.jp2'):  # NIR band
                    nir_path = os.path.join(root, file)
        
        if not swir_path or not nir_path:
            log("Could not find required band files")
            return None
        
        return {
            'swir_path': swir_path,
            'nir_path': nir_path,
            'id': image_info['id'],
            'date': image_info['date']
        }
    
    except Exception as e:
        log(f"Error downloading image: {e}")
        return None

def process_sentinel_image(image_data, grid_cells):
    """Process a Sentinel-2 image and extract features for each grid cell"""
    results = []
    
    # Open the SWIR and NIR bands
    with rasterio.open(image_data['swir_path']) as swir_src, rasterio.open(image_data['nir_path']) as nir_src:
        # Read the bands
        swir = swir_src.read(1)  # SWIR band
        nir = nir_src.read(1)    # NIR band
        
        # Get the transform and dimensions
        transform = swir_src.transform
        height, width = swir.shape
        
        for cell in grid_cells:
            # Get cell bounds
            bounds = cell.bounds
            
            # Transform bounds to image coordinates
            window = transform_bounds(*bounds, transform)
            
            # Convert to pixel coordinates
            col_off = int(window[0])
            row_off = int(window[1])
            col_width = int(window[2] - window[0])
            row_height = int(window[3] - window[1])
            
            # Ensure we don't go out of bounds
            col_off = max(0, min(col_off, width))
            row_off = max(0, min(row_off, height))
            col_width = max(1, min(col_width, width - col_off))
            row_height = max(1, min(row_height, height - row_off))
            
            # Extract data for the cell
            cell_swir = swir[row_off:row_off+row_height, col_off:col_off+col_width]
            cell_nir = nir[row_off:row_off+row_height, col_off:col_off+col_width]
            
            # Skip if the cell is empty
            if cell_swir.size == 0 or cell_nir.size == 0:
                continue
            
            # Calculate features
            humidity = np.mean(cell_swir)
            temperature = np.mean(cell_nir)
            
            # Calculate fire index (simplified version)
            fire_index = (cell_nir - cell_swir) / (cell_nir + cell_swir)
            fire_status = 1 if np.mean(fire_index) > 0.2 else 0
            
            # Calculate distance from reference point
            cell_center = cell.centroid
            distance = haversine_distance(
                cell_center.y, cell_center.x,
                REFERENCE_POINT[1], REFERENCE_POINT[0]
            )
            
            results.append({
                "Date": image_data['date'],
                "Latitude": cell_center.y,
                "Longitude": cell_center.x,
                "Humidity": humidity,
                "Temperature": temperature,
                "Fire": fire_status,
                "DistanceFromReference": distance
            })
    
    return results

def main():
    log("Initializing...")
    
    # Calculate grid size and generate grid cells
    grid_size = calculate_grid_size(aoi_coords, TARGET_CELLS)
    grid_cells = generate_grid(aoi_coords, grid_size)
    
    # Create AOI polygon
    aoi_polygon = Polygon(aoi_coords)
    
    # Process images for each day
    current_date = datetime.strptime(start_date, '%Y-%m-%d')
    end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
    
    all_results = []
    
    while current_date <= end_date_obj:
        date_str = current_date.strftime('%Y-%m-%d')
        log(f"Processing date: {date_str}")
        
        # Search for Sentinel image
        image_info = search_sentinel_images_usgs(date_str, aoi_polygon)
        
        if image_info:
            # Download the image
            image_data = download_sentinel_image_usgs(image_info)
            
            if image_data:
                # Process the image
                results = process_sentinel_image(image_data, grid_cells)
                all_results.extend(results)
                
                # Clean up temporary files
                # Uncomment the following lines if you want to clean up after processing
                # import shutil
                # shutil.rmtree(os.path.join(TEMP_DIR, image_data['id']), ignore_errors=True)
                # os.remove(os.path.join(TEMP_DIR, f"{image_data['id']}.zip"))
            else:
                log(f"Failed to download image for {date_str}")
        else:
            log(f"No suitable image found for {date_str}")
        
        # Move to the next day
        current_date += timedelta(days=1)
        
        # Add a delay to avoid hitting rate limits
        time.sleep(1)
    
    # Save results to CSV
    log("Saving to file...")
    df = pd.DataFrame(all_results)
    df.to_csv(CSV_FILE, index=False)
    log(f"File {CSV_FILE} created.")

if __name__ == "__main__":
    main()
