import ee
import pandas as pd
from datetime import datetime, timedelta
from math import radians, sin, cos, sqrt, atan2
import sys
import os
import requests
from PIL import Image
import io
import concurrent.futures
import threading

ee.Initialize(project='forestfire-pred')

# Увеличиваем область поиска и проверяем координаты
aoi = ee.Geometry.Polygon([
    [[80.107, 50.482], [81.214, 50.335], [81.159, 51.207]]
])

# Расширяем период до летних месяцев 2018-2023
# Все летние месяцы (июнь, июль, август) за 2018-2023 годы
YEARS = [2018, 2019, 2020, 2021, 2022, 2023]
SUMMER_MONTHS = [6, 7, 8]  # Июнь, июль, август

SCALE = 20  # Разрешение SWIR каналов (20 метров)
OUTPUT_DIR = "satellite_images"  # Directory to save images
# Обновляем коллекцию на рекомендуемую версию
COLLECTION = 'COPERNICUS/S2_HARMONIZED'  # Используем актуальную версию Sentinel-2

# Оптимизация и улучшение качества
MAX_IMAGES_PER_YEAR = 90  # 90 изображений для каждого года
MAX_IMAGES_PER_MONTH = 30  # 30 изображений для каждого месяца
MAX_TOTAL_IMAGES = 540  # Максимальное общее количество изображений (90 * 6 лет)
IMAGE_SIZE = 800  # Размер изображения
MAX_WORKERS = 6   # Увеличиваем количество потоков для ускорения

# Блокировка для безопасного вывода в лог
log_lock = threading.Lock()

def log(message):
    with log_lock:
        print(message)
        sys.stdout.flush()

def create_output_dir():
    if not os.path.exists(OUTPUT_DIR):
        os.makedirs(OUTPUT_DIR)
        log(f"Created directory: {OUTPUT_DIR}")

def fetch_images_for_period(start_date, end_date, max_images):
    """Получить список изображений за указанный период"""
    try:
        log(f"Fetching images from {start_date} to {end_date}")
        
        # Используем актуальную версию Sentinel-2
        collection = ee.ImageCollection(COLLECTION) \
            .filterBounds(aoi) \
            .filterDate(start_date, end_date) \
            .filter(ee.Filter.lt('CLOUDY_PIXEL_PERCENTAGE', 30)) \
            .sort('CLOUDY_PIXEL_PERCENTAGE')  # Сортируем по облачности
        
        # Проверяем размер коллекции перед вызовом toList
        count = collection.size().getInfo()
        if count == 0:
            log(f"No images found for period {start_date} to {end_date}")
            return []
        
        # Ограничиваем количество изображений для ускорения
        count_to_get = min(count, max_images)
        log(f"Found {count} images for period {start_date} to {end_date}, processing {count_to_get}")
        
        # Получаем список всех доступных изображений (уже отсортированных по облачности)
        image_list = collection.toList(count_to_get)
        
        available_images = []
        # Получаем информацию о каждом изображении
        for i in range(count_to_get):
            try:
                img = ee.Image(image_list.get(i))
                img_date = img.date().format("yyyy-MM-dd").getInfo()
                
                # Проверяем наличие необходимых каналов
                bands = img.bandNames().getInfo()
                log(f"Image {i+1}/{count_to_get}: Date {img_date}")
                
                # Проверяем наличие нужных каналов
                required_bands = ['B4', 'B8', 'B11', 'B12']
                if all(band in bands for band in required_bands):
                    # Добавляем облачность, если доступна
                    cloud_percent = None
                    try:
                        cloud_percent = img.get('CLOUDY_PIXEL_PERCENTAGE').getInfo()
                    except:
                        pass
                    
                    # Рассчитываем индексы для обнаружения пожаров и сухостоя
                    enhanced_img = calculate_indices(img)
                    
                    available_images.append({
                        'image': enhanced_img,
                        'date': img_date,
                        'cloud_percent': cloud_percent,
                        'bands': bands,
                        'collection': COLLECTION
                    })
                    log(f"  - Added to available images (Cloud: {cloud_percent}%)")
                else:
                    log(f"  - Missing required bands, skipping")
            except Exception as e:
                log(f"Error processing image {i+1}/{count_to_get}: {e}")
        
        return available_images
    except Exception as e:
        log(f"Error fetching images: {e}")
        return []

def fetch_images_for_years(years, months, max_per_year, max_per_month):
    """Получить изображения для нескольких лет"""
    all_images = []
    
    for year in years:
        year_images = []
        for month in months:
            start_date = f"{year}-{month:02d}-01"
            # Определяем последний день месяца
            if month == 6 or month == 8:  # Июнь или август - 30/31 день
                end_date = f"{year}-{month:02d}-{'30' if month == 6 else '31'}"
            else:  # Июль - 31 день
                end_date = f"{year}-{month:02d}-31"
            
            # Получаем изображения за месяц
            month_images = fetch_images_for_period(start_date, end_date, max_per_month)
            year_images.extend(month_images)
        
        # Если за этот год получили больше изображений, чем нужно, оставляем только лучшие
        if len(year_images) > max_per_year:
            # Сортируем по облачности и берем только max_per_year лучших
            year_images.sort(key=lambda x: x['cloud_percent'] if x['cloud_percent'] is not None else 100)
            year_images = year_images[:max_per_year]
        
        log(f"Total images for {year}: {len(year_images)}")
        
        # Создаем подпапку для года
        year_dir = os.path.join(OUTPUT_DIR, str(year))
        if not os.path.exists(year_dir):
            os.makedirs(year_dir)
            log(f"Created directory: {year_dir}")
        
        all_images.extend(year_images)
    
    log(f"Total images to process: {len(all_images)}")
    return all_images

def calculate_indices(image):
    """Рассчитывает индексы для обнаружения пожаров и сухостоя"""
    # Normalized Burn Ratio (NBR) для обнаружения пожаров
    nbr = image.normalizedDifference(['B8', 'B12']).rename('NBR')
    
    # Normalized Difference Vegetation Index (NDVI) для анализа растительности
    ndvi = image.normalizedDifference(['B8', 'B4']).rename('NDVI')
    
    # Добавляем индексы к изображению
    enhanced_image = image.addBands(nbr).addBands(ndvi)
    
    return enhanced_image

def save_image(image_data, vis_type, file_prefix):
    """Сохранение одного типа изображения"""
    try:
        image = image_data['image']
        date_str = image_data['date']
        year = date_str.split('-')[0]  # Извлекаем год из даты
        
        # Различные типы визуализации для разных целей
        if vis_type == 'fire':
            # SWIR2 (B12) / NIR (B8) - хорошо показывает пожары
            rgb_image = image.select(['B12', 'B8'], ['red', 'green'])
            rgb_image = rgb_image.addBands(rgb_image.select(['green']).rename('blue'))
            
            # Улучшенные настройки контрастности для выделения пожаров
            url = rgb_image.getThumbURL({
                'region': aoi,
                'dimensions': IMAGE_SIZE,
                'format': 'png',
                'min': 0,
                'max': 3000,
                'gamma': 1.2  # Усиливаем контраст
            })
        elif vis_type == 'drought':
            # SWIR1 (B11) / NIR (B8) - хорошо показывает сухостоя
            rgb_image = image.select(['B11', 'B8'], ['red', 'green'])
            rgb_image = rgb_image.addBands(rgb_image.select(['green']).rename('blue'))
            
            # Улучшенные настройки для выделения сухих участков
            url = rgb_image.getThumbURL({
                'region': aoi,
                'dimensions': IMAGE_SIZE,
                'format': 'png',
                'min': 0,
                'max': 3500,
                'gamma': 1.1
            })
        elif vis_type == 'nbr':
            # Normalized Burn Ratio - специально для обнаружения гарей
            # Используем цветовую палитру от зеленого (здоровая растительность) к красному (гарь)
            nbr_palette = ['#1a9850', '#91cf60', '#d9ef8b', '#ffffbf', '#fee08b', '#fc8d59', '#d73027']
            
            url = image.select('NBR').getThumbURL({
                'region': aoi,
                'dimensions': IMAGE_SIZE,
                'format': 'png',
                'min': -0.5,
                'max': 1,
                'palette': nbr_palette
            })
        elif vis_type == 'natural':
            # Естественная цветовая композиция (RGB)
            rgb_image = image.select(['B4', 'B3', 'B2'], ['red', 'green', 'blue'])
            
            url = rgb_image.getThumbURL({
                'region': aoi,
                'dimensions': IMAGE_SIZE,
                'format': 'png',
                'min': 0,
                'max': 3000,
                'gamma': 1.2
            })
        else:
            log(f"Unknown visualization type: {vis_type}")
            return False
        
        # Скачиваем и сохраняем изображение
        response = requests.get(url)
        if response.status_code == 200:
            cloud_info = ""
            if image_data['cloud_percent'] is not None:
                cloud_info = f"_cloud{image_data['cloud_percent']:.1f}"
            
            # Сохраняем в подпапку соответствующего года
            year_dir = os.path.join(OUTPUT_DIR, year)    
            img_path = os.path.join(year_dir, f"{file_prefix}_{date_str}{cloud_info}.png")
            with open(img_path, 'wb') as f:
                f.write(response.content)
            log(f"Saved {file_prefix} image: {img_path}")
            return True
        else:
            log(f"Failed to download {file_prefix} image for {date_str}")
            return False
    except Exception as e:
        log(f"Error saving {file_prefix} image for {date_str}: {e}")
        return False

def process_image(img_data, idx, total):
    """Обработка одного изображения в отдельном потоке"""
    log(f"Processing image {idx+1}/{total}: {img_data['date']}")
    try:
        # Сохраняем все типы визуализации
        save_image(img_data, 'fire', 'fire')
        save_image(img_data, 'drought', 'drought')
        save_image(img_data, 'nbr', 'nbr')
        save_image(img_data, 'natural', 'natural')
        
        return True
    except Exception as e:
        log(f"Error processing image {idx+1}/{total}: {e}")
        return False

def process_available_images():
    # Получаем изображения за все летние месяцы 2018-2023
    available_images = fetch_images_for_years(YEARS, SUMMER_MONTHS, MAX_IMAGES_PER_YEAR, MAX_IMAGES_PER_MONTH)
    
    if not available_images:
        log("No valid images found for processing")
        return
    
    log(f"Processing {len(available_images)} available images using {MAX_WORKERS} parallel workers")
    
    # Используем ThreadPoolExecutor для параллельной обработки
    with concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        # Создаем задачи для каждого изображения
        futures = []
        for i, img_data in enumerate(available_images):
            future = executor.submit(process_image, img_data, i, len(available_images))
            futures.append(future)
        
        # Ждем завершения всех задач
        for future in concurrent.futures.as_completed(futures):
            # Здесь можно обработать результат, если нужно
            pass

# Create output directory
create_output_dir()

# Process and save images
log(f"Starting processing with {MAX_IMAGES_PER_YEAR} images per year (max {MAX_IMAGES_PER_MONTH} per month), {MAX_WORKERS} workers, image size {IMAGE_SIZE}px")
process_available_images()

log("Image processing completed.")
