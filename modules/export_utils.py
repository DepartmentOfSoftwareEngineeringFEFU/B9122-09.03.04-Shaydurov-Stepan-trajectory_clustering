"""
Модуль для экспорта данных
"""

import pandas as pd
import json
import os
import tempfile
from datetime import datetime
from typing import Optional
try:
    from selenium import webdriver
    from selenium.webdriver.chrome.options import Options
    from PIL import Image
    SELENIUM_AVAILABLE = True
except ImportError:
    SELENIUM_AVAILABLE = False
    print("⚠️ Selenium не установлен. Экспорт карт в PNG недоступен.")


def export_map_to_png_bytes(map_object, width: int = 2560, height: int = 1744) -> Optional[bytes]:
    """
    Экспорт карты в PNG байты для скачивания
    """
    if not SELENIUM_AVAILABLE:
        return None

    if map_object is None:
        return None

    # Временный HTML файл
    with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8') as f:
        map_object.save(f.name)
        html_path = f.name

    driver = None
    try:
        chrome_options = Options()
        chrome_options.add_argument("--headless")
        chrome_options.add_argument("--no-sandbox")
        chrome_options.add_argument("--disable-dev-shm-usage")
        chrome_options.add_argument(f"--window-size={width},{height}")

        driver = webdriver.Chrome(options=chrome_options)
        driver.get(f"file://{html_path}")

        # Скриншот
        screenshot_path = html_path.replace('.html', '.png')
        driver.save_screenshot(screenshot_path)

        # Читаем в байты
        with open(screenshot_path, 'rb') as img_file:
            png_bytes = img_file.read()

        return png_bytes

    except Exception as e:
        print(f"Ошибка: {e}")
        return None

    finally:
        if driver:
            driver.quit()
        # Удаляем временные файлы
        try:
            os.unlink(html_path)
            if os.path.exists(screenshot_path):
                os.unlink(screenshot_path)
        except:
            pass



def is_png_export_available() -> bool:
    """Проверка доступности экспорта в PNG"""
    return SELENIUM_AVAILABLE

def export_to_geojson(df: pd.DataFrame, filename: str = "trajectories.geojson") -> str:
    """
    Экспорт DataFrame в GeoJSON формат
    """
    if df is None or df.empty:
        print("⚠️ Нет данных для экспорта")
        return None

    features = []

    for _, row in df.iterrows():
        if row.get('lat') is None or row.get('lon') is None:
            continue

        properties = {}
        for col in df.columns:
            if col not in ['lat', 'lon']:
                value = row[col]
                if hasattr(value, 'item'):
                    value = value.item()
                elif isinstance(value, pd.Timestamp):
                    value = value.isoformat()
                properties[col] = value

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [float(row['lon']), float(row['lat'])]
            },
            "properties": properties
        }
        features.append(feature)

    geojson = {
        "type": "FeatureCollection",
        "features": features
    }

    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(geojson, f, ensure_ascii=False, indent=2)

    print(f"✅ Экспортировано {len(features)} точек в {filename}")
    return filename


def export_to_csv(df: pd.DataFrame, filename: str = "trajectories.csv") -> str:
    """Экспорт DataFrame в CSV"""
    if df is None or df.empty:
        return None
    df.to_csv(filename, index=False, encoding='utf-8')
    return filename