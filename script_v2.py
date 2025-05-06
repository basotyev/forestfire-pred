import ee
import sys
import os
import requests
from PIL import Image
import io
import threading
import numpy as np

# Инициализация Earth Engine
ee.Initialize(project='forestfire-pred')

# Константы
OUTPUT_DIR = "satellite_images"
COLLECTION = 'COPERNICUS/S2_HARMONIZED'
IMAGE_SIZE = 800
SCALE = 20

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
SUMMER_MONTHS = [6, 7, 8]
MAX_IMAGES_PER_YEAR = 90
MAX_IMAGES_PER_MONTH = 30

aoi = ee.Geometry.Polygon([
    [[80.107, 50.482], [81.214, 50.335], [81.159, 51.207]]
])

log_lock = threading.Lock()

def log(message):
    with log_lock:
        print(message)
        sys.stdout.flush()

def crop_black_border_pil(pil_img, threshold=10):
    np_img = np.array(pil_img.convert("L"))
    mask = np_img > threshold
    coords = np.argwhere(mask)
    if coords.size == 0:
        return pil_img
    y0, x0 = coords.min(axis=0)
    y1, x1 = coords.max(axis=0) + 1
    return pil_img.crop((x0, y0, x1, y1))

def calculate_indices(image):
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    return image.addBands([nbr, ndvi, ndwi])

def fetch_images_for_period(start_date, end_date, max_images):
    log(f"Fetching images from {start_date} to {end_date}")
    try:
        collection = ee.ImageCollection(COLLECTION) \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')

        count = collection.size().getInfo()
        if count == 0:
            log(f"No images found for {start_date} to {end_date}")
            return []

        image_list = collection.toList(min(count, max_images))
        available_images = []

        for i in range(min(count, max_images)):
            try:
                img = ee.Image(image_list.get(i))
                img_date = img.date().format("yyyy-MM-dd").getInfo()
                bands = img.bandNames().getInfo()

                if all(b in bands for b in ['B4', 'B8', 'B11', 'B12']):
                    cloud_percent = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                    enhanced_img = calculate_indices(img)
                    available_images.append({
                        'image': enhanced_img,
                        'date': img_date,
                        'cloud_percent': cloud_percent
                    })
            except Exception as e:
                log(f"Error processing image {i+1}: {e}")

        return available_images
    except Exception as e:
        log(f"Error fetching images: {e}")
        return []

def fetch_images_for_years(years, months):
    all_images = []
    for year in years:
        year_images = []
        for month in months:
            start_date = f"{year}-{month:02d}-01"
            end_date = f"{year}-{month:02d}-30" if month == 6 else f"{year}-{month:02d}-31"
            month_images = fetch_images_for_period(start_date, end_date, MAX_IMAGES_PER_MONTH)
            year_images.extend(month_images)
        year_images.sort(key=lambda x: x['cloud_percent'] if x['cloud_percent'] is not None else 100)
        all_images.extend(year_images[:MAX_IMAGES_PER_YEAR])
    return all_images

def save_image(image_data, vis_type):
    image = image_data['image']
    date_str = image_data['date']
    year = date_str.split('-')[0]
    vis_url = None

    if vis_type == 'fire':
        rgb = image.select(['B12', 'B8'], ['red', 'green'])
        rgb = rgb.addBands(rgb.select('green').rename('blue'))
        vis_url = rgb.getThumbURL({'region': aoi, 'dimensions': IMAGE_SIZE, 'format': 'png', 'min': 0, 'max': 3000, 'gamma': 1.2})

    elif vis_type == 'drought':
        rgb = image.select(['B11', 'B8'], ['red', 'green'])
        rgb = rgb.addBands(rgb.select('green').rename('blue'))
        vis_url = rgb.getThumbURL({'region': aoi, 'dimensions': IMAGE_SIZE, 'format': 'png', 'min': 0, 'max': 3500, 'gamma': 1.1})

    elif vis_type == 'nbr':
        nbr = image.select('NBR')
        ndwi = image.select('NDWI')
        masked_nbr = nbr.updateMask(ndwi.lt(0.3))
        palette = ['#1a9850', '#91cf60', '#d9ef8b', '#ffffbf', '#fee08b', '#fc8d59', '#d73027']
        vis_url = masked_nbr.getThumbURL({'region': aoi, 'dimensions': IMAGE_SIZE, 'format': 'png', 'min': -0.5, 'max': 1, 'palette': palette})

    elif vis_type == 'natural':
        rgb = image.select(['B4', 'B3', 'B2'], ['red', 'green', 'blue'])
        vis_url = rgb.getThumbURL({'region': aoi, 'dimensions': IMAGE_SIZE, 'format': 'png', 'min': 0, 'max': 3000, 'gamma': 1.2})

    else:
        log(f"Unknown visualization: {vis_type}")
        return

    try:
        response = requests.get(vis_url)
        if response.status_code == 200:
            out_dir = os.path.join(OUTPUT_DIR, year)
            os.makedirs(out_dir, exist_ok=True)
            path = os.path.join(out_dir, f"{vis_type}_{date_str}.png")
            img = Image.open(io.BytesIO(response.content))
            crop_black_border_pil(img).save(path)
            log(f"Saved {vis_type}: {path}")
        else:
            log(f"Failed to download {vis_type} for {date_str}")
    except Exception as e:
        log(f"Error saving {vis_type} for {date_str}: {e}")

def process_image(img_data, idx, total):
    log(f"Processing image {idx+1}/{total}: {img_data['date']}")
    for vis in ['fire', 'drought', 'nbr', 'natural']:
        save_image(img_data, vis)

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    log("Fetching and processing images...")
    images = fetch_images_for_years(YEARS, SUMMER_MONTHS)
    for idx, img_data in enumerate(images):
        process_image(img_data, idx, len(images))
    log("All done.")

if __name__ == '__main__':
    main()