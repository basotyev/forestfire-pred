import os
import ee
import cv2
import io
import pickle
import requests
import hashlib
import numpy as np
import pandas as pd
from PIL import Image
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler

# --- Инициализация Earth Engine ---
ee.Initialize(project='forestfire-pred')

IMAGE_SIZE = 800
VIS_TYPES = ['fire', 'drought', 'nbr', 'natural']
DATA_DIR = "main_output"


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
    return image.addBands(nbr).addBands(ndvi)


def get_images_for_location(place_coords, year, month, day):
    aoi = ee.Geometry.Polygon([place_coords])
    base_date = datetime(year, month, day)

    for offset in range(-2, 3):
        try_date = base_date + timedelta(days=offset)
        date_str = try_date.strftime('%Y-%m-%d')

        collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_str, (try_date + timedelta(days=1)).strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 40)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')

        size = collection.size().getInfo()
        if size > 0:
            image = ee.Image(collection.first())
            image = calculate_indices(image)
            actual_date = image.date().format("yyyy-MM-dd").getInfo()
            return image, actual_date, aoi
    raise Exception(f"No suitable image found for {base_date.strftime('%Y-%m-%d')} ±2 days")


def download_and_save(image, date_str, vis_type, out_dir, aoi):
    ndwi = image.normalizedDifference(['B3', 'B8']).rename('NDWI')
    water_mask = ndwi.lt(0.3)
    if vis_type == 'fire':
        rgb = image.select(['B12', 'B8'], ['red', 'green'])
        rgb = rgb.addBands(rgb.select(['green']).rename('blue'))
        vis_params = {'min': 0, 'max': 3000, 'gamma': 1.2}
    elif vis_type == 'drought':
        rgb = image.select(['B11', 'B8'], ['red', 'green'])
        rgb = rgb.addBands(rgb.select(['green']).rename('blue'))
        vis_params = {'min': 0, 'max': 3500, 'gamma': 1.1}
    elif vis_type == 'nbr':
        nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        rgb = nbr.updateMask(water_mask)
        vis_params = {
            'min': -0.5, 'max': 1,
            'palette': ['#1a9850', '#91cf60', '#d9ef8b', '#ffffbf',
                        '#fee08b', '#fc8d59', '#d73027']
        }
    elif vis_type == 'natural':
        rgb = image.select(['B4', 'B3', 'B2'], ['red', 'green', 'blue'])
        vis_params = {'min': 0, 'max': 3000, 'gamma': 1.2}
    else:
        print(f"Unknown visualization: {vis_type}")
        return

    vis_params.update({
        'region': aoi,
        'dimensions': IMAGE_SIZE,
        'format': 'png'
    })

    path = os.path.join(out_dir, f"{vis_type}_{date_str}.png")
    if os.path.exists(path):
        return

    url = rgb.getThumbURL(vis_params)
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        cropped = crop_black_border_pil(img)
        cropped.save(path)
        print(f"Saved {vis_type} image → {path}")
    else:
        print(f"Failed to download {vis_type} image for {date_str}")


def download_all_visualizations(image, date_str, aoi, place_name):
    out_dir = os.path.join(DATA_DIR, place_name)
    os.makedirs(out_dir, exist_ok=True)
    for vis in VIS_TYPES:
        download_and_save(image, date_str, vis, out_dir, aoi)


def load_image(file_path):
    img = cv2.imread(file_path)
    return img if img is not None else None


def extract_features(image):
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    bgr_means = np.mean(image, axis=(0, 1))
    hsv_means = np.mean(hsv, axis=(0, 1))
    bgr_std = np.std(image, axis=(0, 1))
    hsv_std = np.std(hsv, axis=(0, 1))
    b, g, r = cv2.split(image)
    h, s, v = cv2.split(hsv)
    histograms = np.concatenate([
        cv2.calcHist([b], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([g], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([r], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([h], [0], None, [16], [0, 180]).flatten(),
        cv2.calcHist([s], [0], None, [16], [0, 256]).flatten(),
        cv2.calcHist([v], [0], None, [16], [0, 256]).flatten(),
    ])
    histograms /= np.sum(histograms)
    pixels = image.reshape(-1, 3).astype(np.float32)
    _, _, centers = cv2.kmeans(pixels, 5, None,
                               (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2),
                               10, cv2.KMEANS_RANDOM_CENTERS)
    red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) + \
               cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
    red_percentage = np.sum(red_mask > 0) / image.size
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    bright_mask = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
    bright_percentage = np.sum(bright_mask > 0) / image.size

    return np.concatenate([
        bgr_means, bgr_std, hsv_means, hsv_std,
        centers.flatten(), histograms,
        [red_percentage, bright_percentage]
    ])


def process_location_set(locations):
    results = []
    models = {}
    for model_name in ['rf', 'svm']:
        with open(f"results_{model_name}/model.pkl", "rb") as f:
            models[model_name] = pickle.load(f)

    for idx, (place_name, coords, year, month, day) in enumerate(locations):
        print(f"[{idx + 1}] {place_name} ({year}-{month:02d}-{day:02d})")
        try:
            image, actual_date, aoi = get_images_for_location(coords, year, month, day)
            download_all_visualizations(image, actual_date, aoi, place_name)
        except Exception as e:
            print(f"download error: {e}")
            continue

        feat_vec = []
        skip = False
        for vis in VIS_TYPES:
            path = os.path.join(DATA_DIR, place_name, f"{vis}_{actual_date}.png")
            img = load_image(path)
            if img is None:
                skip = True
                break
            feats = extract_features(img)
            feat_vec.extend(feats)

        if skip:
            print(f"⚠️ Skipped: {place_name} (incorrect images)")
            continue

        for model_name, (model, scaler) in models.items():
            scaled = scaler.transform([feat_vec])
            prob = model.predict_proba(scaled)[0][1]
            pred = model.predict(scaled)[0]
            results.append({
                "place": place_name,
                "requested_date": f"{year}-{month:02d}-{day:02d}",
                "actual_date": actual_date,
                "model": model_name,
                "prediction": int(pred),
                "probability": round(prob, 4)
            })
    return pd.DataFrame(results)


def main():
    locations = [
        ("region_1", [[80.1, 50.4], [81.2, 50.3], [81.1, 51.2]], 2023, 7, 15),
        ("region_2", [[79.5, 49.9], [80.6, 50.0], [80.0, 50.7]], 2023, 8, 10),
        ("region_3", [[78.2, 48.9], [79.3, 49.0], [78.8, 49.7]], 2023, 6, 20),
    ]

    df = process_location_set(locations)
    print(df)
    df.to_csv(DATA_DIR + "/fire_predictions_summary.csv", index=False)


if __name__ == "__main__":
    main()
