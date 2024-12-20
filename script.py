import ee
import pandas as pd
from datetime import datetime

ee.Initialize(project='forestfire-pred')

aoi = ee.Geometry.Polygon([
    [[80.0, 49.0], [86.0, 49.0], [86.0, 47.0], [80.0, 47.0], [80.0, 49.0]]
])

start_date = '2023-04-01'
end_date = '2023-09-30'

collection = ee.ImageCollection('COPERNICUS/S2') \
    .filterBounds(aoi) \
    .filterDate(start_date, end_date) \
    .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 20)) \
    .limit(50)

GRID_SIZE = 0.2
SCALE = 200
CSV_FILE = "processed_data_2023.csv"

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

def process_images(collection, grid_cells):
    data = []
    images = collection.toList(collection.size()).getInfo()

    for cell_index, cell in enumerate(grid_cells):
        cell_center = cell.centroid().coordinates().getInfo()
        lat, lon = cell_center[1], cell_center[0]
        print(f"[{cell_index + 1}/{len(grid_cells)}] Обработка ячейки: {lat}, {lon}...")

        for image_info in images:
            try:
                img = ee.Image(image_info['id'])
                timestamp = image_info.get('properties', {}).get('system:time_start', None)
                date = datetime.utcfromtimestamp(timestamp / 1000).strftime('%Y-%m-%d') if timestamp else "Unknown"
                stats = img.reduceRegion(
                    reducer=ee.Reducer.mean(),
                    geometry=cell,
                    scale=SCALE,
                    bestEffort=True
                ).getInfo()

                humidity = stats.get('B11', None)  # SWIR (примерный индикатор влажности)
                temperature = stats.get('B12', None)  # SWIR2 (примерный индикатор температуры)
                fire_index = stats.get('B8', None)  # NIR (примерный индикатор пожара)

                if humidity and temperature:
                    data.append({
                        "Date": date,
                        "Latitude": lat,
                        "Longitude": lon,
                        "Humidity": humidity,
                        "Temperature": temperature,
                        "Fire": 1 if fire_index and fire_index > 1500 else 0
                    })

            except Exception as e:
                print(f"  Ошибка при обработке изображения: {e}")
                continue

    return data

print("Генерация сетки...")
grid_cells = generate_grid(aoi, GRID_SIZE)

print("Обработка изображений...")
processed_data = process_images(collection, grid_cells)

print("Сохранение в файл...")
df = pd.DataFrame(processed_data)
df.to_csv(CSV_FILE, index=False)

print(f"Файл {CSV_FILE} создан.")
