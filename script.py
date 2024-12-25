import ee
import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import sys

ee.Initialize(project='forestfire-pred')

aoi = ee.Geometry.Polygon([
    [[78.58013561221814,51.28376604439698], [78.20660045596814,50.994253517722704], [78.79986217471814,50.3537270276519], [78.40435436221814,50.04429404493862], [77.65728404971814,50.25548861230339], [76.86626842471814,49.44798023538978], [77.63531139346814,48.68500731531969], [78.62408092471814,47.93974579613136], [80.38189342471814,48.1454115464798], [81.87603404971814,49.57637905477649], [81.39263561221814,50.17112270356419], [81.59038951846814,50.63329091010904], [80.77740123721814,51.15991154397452], [80.11822154971814,50.66115634184031], [78.58013561221814,51.28376604439698]]
])


REFERENCE_POINT = [80.23, 50.42]

start_date = '2023-04-01'
end_date = '2023-09-30'

SCALE = 200  # Processing scale in meters
CSV_FILE = "processed_data_2023.csv"
TARGET_CELLS = 100  # Desired number of grid cells


def log(message):
    print(message)
    sys.stdout.flush()

def calculate_grid_size(aoi, target_cells):
    coords = aoi.coordinates().get(0).getInfo()
    min_lon = min(coord[0] for coord in coords)
    max_lon = max(coord[0] for coord in coords)
    min_lat = min(coord[1] for coord in coords)
    max_lat = max(coord[1] for coord in coords)

    lon_range = max_lon - min_lon
    lat_range = max_lat - min_lat

    cell_area = (lon_range * lat_range) / target_cells
    grid_size = cell_area ** 0.5
    return grid_size

GRID_SIZE = calculate_grid_size(aoi, TARGET_CELLS)


def generate_grid(aoi, grid_size):
    coords = aoi.coordinates().get(0).getInfo()
    min_lon = min(coord[0] for coord in coords)
    max_lon = max(coord[0] for coord in coords)
    min_lat = min(coord[1] for coord in coords)
    max_lat = max(coord[1] for coord in coords)

    grid_cells = []
    lat = min_lat
    while lat < max_lat:
        lon = min_lon
        while lon < max_lon:
            grid_cells.append(ee.Geometry.Rectangle([lon, lat, lon + grid_size, lat + grid_size]))
            lon += grid_size
        lat += grid_size

    return grid_cells


grid_cells = generate_grid(aoi, GRID_SIZE)

def haversine_distance(lat1, lon1, lat2, lon2):
    R = 6371.0  # Earth radius in kilometers
    dlat = radians(lat2 - lat1)
    dlon = radians(lon2 - lon1)
    a = sin(dlat / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(dlon / 2) ** 2
    c = 2 * atan2(sqrt(a), sqrt(1 - a))
    return R * c

def fetch_image_by_day(date):
    collection = ee.ImageCollection('COPERNICUS/S2_SR_HARMONIZED') \
        .filterBounds(aoi) \
        .filterDate(date, date.advance(1, 'day')) \
        .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
        .sort('CLOUDY_PIXEL_PERCENTAGE')

    image = collection.first()
    return image if image is not None else None

def process_images_by_day(start_date, end_date, grid_cells):
    results = []
    start_date = datetime.strptime(start_date, "%Y-%m-%d")
    end_date = datetime.strptime(end_date, "%Y-%m-%d")

    current_date = start_date
    while current_date <= end_date:
        day_str = current_date.strftime('%Y-%m-%d')
        log(f"Processing date: {day_str}")

        img = fetch_image_by_day(ee.Date(day_str))
        if img is None:
            log(f"No image available for {day_str}")
            current_date += timedelta(days=1)
            continue

        try:
            img_date = img.date().format("yyyy-MM-dd").getInfo()

            stats = img.reduceRegions(
                collection=ee.FeatureCollection(grid_cells),
                reducer=ee.Reducer.mean(),
                scale=SCALE
            ).getInfo()

            for stat in stats['features']:
                cell_geometry = stat['geometry']
                properties = stat['properties']

                lon, lat = cell_geometry['coordinates'][0][0]  # Assuming simple polygons
                closest_distance = haversine_distance(lat, lon, REFERENCE_POINT[1], REFERENCE_POINT[0])

                humidity = properties.get('B11', None)
                temperature = properties.get('B12', None)
                fire_index = properties.get('B8', None)

                fire_status = 0  # Default: No fire

                if fire_index is not None and fire_index > 0:  # Ensure fire_index is valid
                    fire_index_normalized = fire_index / 10000  # Normalize to reflectance
                    log(f"Raw Fire Index: {fire_index}, Normalized: {fire_index_normalized}")

                    if 0 <= fire_index_normalized <= 1:  # Valid reflectance range
                        if fire_index_normalized > 0.2:  # Adjust threshold based on analysis
                            fire_status = 1
                        else:
                            log(f"Fire index {fire_index_normalized} below threshold.")
                    else:
                        log(f"Invalid normalized fire index: {fire_index_normalized}")
                else:
                    log("Fire index is None or invalid.")

                if humidity and temperature:
                    results.append({
                        "Date": img_date,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Humidity": humidity,
                        "Temperature": temperature,
                        "Fire": fire_status,
                        "DistanceFromReference": closest_distance
                    })

        except Exception as e:
            log(f"Error processing image for {day_str}: {e}")

        current_date += timedelta(days=1)

    return results


log("Processing images by day...")
processed_data = process_images_by_day(start_date, end_date, grid_cells)

log("Saving to file...")
df = pd.DataFrame(processed_data)
df.to_csv(CSV_FILE, index=False)

log(f"File {CSV_FILE} created.")
