import os
import io
import ee
import cv2
import json
import pickle
import numpy as np
import requests
from PIL import Image
from flask import Flask, request, jsonify
from datetime import datetime, timedelta

ee.Initialize(project='forestfire-pred')

app = Flask(__name__)
IMAGE_SIZE = 800
DATA_DIR = "main_output"
VIS_TYPES = ['fire', 'drought', 'nbr', 'natural']

# Загрузка только модели RF
with open("results_rf/model.pkl", "rb") as f:
    RF_MODEL = pickle.load(f)

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
    return image.addBands(image.normalizedDifference(['B8', 'B12']).rename('NBR')) \
                .addBands(image.normalizedDifference(['B8', 'B4']).rename('NDVI'))

def fetch_satellite_image(aoi, target_date):
    for offset in range(-2, 3):
        date_try = target_date + timedelta(days=offset)
        date_str = date_try.strftime('%Y-%m-%d')
        collection = ee.ImageCollection('COPERNICUS/S2_HARMONIZED') \
            .filterBounds(aoi) \
            .filterDate(date_str, (date_try + timedelta(days=1)).strftime('%Y-%m-%d')) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 100)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')
        if collection.size().getInfo() > 0:
            image = ee.Image(collection.first())
            return calculate_indices(image), date_str
    raise Exception("No suitable image found ±2 days from requested date")

def download_image(image, vis_type, aoi, date_str, output_dir):
    vis_map = {
        'fire': (['B12', 'B8'], 3000),
        'drought': (['B11', 'B8'], 3500),
        'natural': (['B4', 'B3', 'B2'], 3000)
    }

    if vis_type in vis_map:
        bands, vmax = vis_map[vis_type]
        rgb = image.select(bands[:2], ['red', 'green'])
        rgb = rgb.addBands(rgb.select(['green']).rename('blue'))
        vis_params = {'min': 0, 'max': vmax, 'gamma': 1.2}
    elif vis_type == 'nbr':
        rgb = image.normalizedDifference(['B8', 'B12']).rename('NBR')
        vis_params = {
            'min': -0.5, 'max': 1,
            'palette': ['#1a9850', '#91cf60', '#d9ef8b', '#ffffbf',
                        '#fee08b', '#fc8d59', '#d73027']
        }
    else:
        return None

    vis_params.update({
        'region': aoi,
        'dimensions': IMAGE_SIZE,
        'format': 'png'
    })

    url = rgb.getThumbURL(vis_params)
    response = requests.get(url)
    if response.status_code == 200:
        img = Image.open(io.BytesIO(response.content))
        img = crop_black_border_pil(img)
        path = os.path.join(output_dir, f"{vis_type}_{date_str}.png")
        img.save(path)
        return path
    return None

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

@app.route('/predict', methods=['POST'])
def predict_fire():


    try:
        data = request.get_json()
        coords = data.get("aoi")
        date = data.get("date")
        place = data.get("place", "anonymous")
        target_date = datetime.strptime(date, "%Y-%m-%d")
        aoi = ee.Geometry.Polygon([coords]) if not isinstance(coords[0][0], list) else ee.Geometry.Polygon(coords)
        image, real_date = fetch_satellite_image(aoi, target_date)
        out_dir = os.path.join(DATA_DIR, place)
        os.makedirs(out_dir, exist_ok=True)

        features = []
        for vis in VIS_TYPES:
            path = download_image(image, vis, aoi, real_date, out_dir)
            if path is None:
                return jsonify({"error": f"Failed to download {vis} visualization"}), 500
            img = cv2.imread(path)
            if img is None:
                return jsonify({"error": f"Failed to load image {path}"}), 500
            features.extend(extract_features(img))

        prob = float(RF_MODEL.predict_proba([features])[0][1])
        pred = int(RF_MODEL.predict([features])[0])

        return jsonify({
            "place": place,
            "requested_date": date,
            "image_date": real_date,
            "prediction": pred,
            "probability": round(prob, 4)
        })

    except Exception as e:
        import traceback
        print("Error during /predict:", traceback.format_exc())
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8081)
