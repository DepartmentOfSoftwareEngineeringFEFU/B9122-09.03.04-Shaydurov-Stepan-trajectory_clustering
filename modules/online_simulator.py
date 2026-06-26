"""
Модуль онлайн-имитации судов для Дальнего Востока
Фиксированные суда с заданными траекториями (без суши)
"""

import time
import math
import pandas as pd
import threading
from queue import Queue
from typing import Dict, Optional


class OnlineSimulator:
    """Генератор имитации судов с заданными маршрутами"""

    def __init__(self):
        self.running = False
        self.thread = None
        self.queue = Queue()
        self.vessels = {}
        self.update_interval = 2.0

    def start_simulation(self, num_vessels: int = 4, update_interval: float = 2.0):
        """
        Запуск симуляции с фиксированными судами

        Параметры:
        - num_vessels: игнорируется (всегда 4 судна)
        - update_interval: интервал обновления позиций
        """
        self.running = True
        self.update_interval = update_interval

        # ========== ОПРЕДЕЛЕНИЕ СУДОВ ==========

        # 1. Судно из Владивостока (движется на юго-восток)
        vessel_vladivostok = {
            "mmsi": 273210001,
            "lat": 42.9,
            "lon": 132.1,
            "sog": 15.0,      # скорость 15 узлов
            "cog": 135.0,     # курс 135° (юго-восток)
            "type": "Контейнеровоз",
            "source": "ONLINE_SIM",
            "start_point": "vladivostok"
        }

        # 2. Судно из Сахалина (движется на запад)
        vessel_sakhalin = {
            "mmsi": 273210002,
            "lat": 46.6,
            "lon": 142.8,
            "sog": 12.0,      # скорость 12 узлов
            "cog": 270.0,     # курс 270° (запад)
            "type": "Танкер",
            "source": "ONLINE_SIM",
            "start_point": "sakhalin"
        }

        # 3. Судно в открытом море (движется на север)
        vessel_open_sea = {
            "mmsi": 273210003,
            "lat": 48.0,
            "lon": 145.0,
            "sog": 18.0,      # скорость 18 узлов
            "cog": 0.0,       # курс 0° (север)
            "type": "Грузовое",
            "source": "ONLINE_SIM",
            "start_point": "open_sea"
        }

        # 4. Стоячее судно (шум) — не двигается
        vessel_stationary = {
            "mmsi": 273210004,
            "lat": 42.0,
            "lon": 131.0,
            "sog": 0.0,       # скорость 0 (стоит)
            "cog": 0.0,
            "type": "Рыболовное",
            "source": "ONLINE_SIM",
            "start_point": "stationary"
        }

        # 4. Стоячее судно (шум) — не двигается
        vessel_open_sea2 = {
            "mmsi": 273210005,
            "lat": 38.9,
            "lon": 134.2,
            "sog": 20.0,  # скорость 0 (стоит)
            "cog": 3.0,
            "type": "Контейнеровоз",
            "source": "ONLINE_SIM",
            "start_point": "stationary"
        }


        self.vessels = {
            273210001: vessel_vladivostok,
            273210002: vessel_sakhalin,
            273210003: vessel_open_sea,
            273210004: vessel_stationary,
            273210005: vessel_open_sea2
        }

        def simulate():
            print(f"🔄 Симуляция запущена: 5 судна, интервал {update_interval} сек")
            print("   🚢 Владивосток → движется на юго-восток")
            print("   🚢 Сахалин → движется на запад")
            print("   🚢 Открытое море → движется на север")
            print("   🚢 Стоячее судно (шум)")

            # Отправляем начальные позиции
            for mmsi, vessel in self.vessels.items():
                packet = {
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3],
                    'mmsi': vessel['mmsi'],
                    'lat': vessel['lat'],
                    'lon': vessel['lon'],
                    'sog': vessel['sog'],
                    'cog': vessel['cog'],
                    'type': vessel['type'],
                    'source': 'ONLINE_SIM'
                }
                self.queue.put(packet)
                print(f"📡 Судно {vessel['mmsi']}: старт ({vessel['lat']}, {vessel['lon']})")

            while self.running:
                time.sleep(update_interval)

                # Обновляем позиции движущихся судов
                for mmsi, vessel in self.vessels.items():
                    # Стоячее судно не двигается
                    if vessel['sog'] == 0:
                        continue

                    # Преобразуем курс в радианы
                    cog_rad = vessel['cog'] * math.pi / 180
                    # Скорость в градусах (1 узел ≈ 0.00001 градуса в секунду)
                    speed_deg = vessel['sog'] / 100000 * self.update_interval

                    # Обновляем координаты
                    new_lat = vessel['lat'] + speed_deg * math.cos(cog_rad)
                    new_lon = vessel['lon'] + speed_deg * math.sin(cog_rad)

                    # Обновляем судно
                    vessel['lat'] = round(new_lat, 6)
                    vessel['lon'] = round(new_lon, 6)

                    # Отправляем обновление
                    packet = {
                        'timestamp': pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3],
                        'mmsi': vessel['mmsi'],
                        'lat': vessel['lat'],
                        'lon': vessel['lon'],
                        'sog': vessel['sog'],
                        'cog': vessel['cog'],
                        'type': vessel['type'],
                        'source': 'ONLINE_SIM'
                    }
                    self.queue.put(packet)

                    # Логируем движение (только иногда)
                    if int(time.time()) % 10 < 2:
                        print(f"🚢 Судно {mmsi}: позиция ({vessel['lat']:.4f}, {vessel['lon']:.4f})")

            print("⏹️ Симуляция остановлена")

        self.thread = threading.Thread(target=simulate, daemon=True)
        self.thread.start()

    def update_vessel_params(self, mmsi: int, lat: float = None, lon: float = None,
                             sog: float = None, cog: float = None):
        """Обновление параметров судна (можно вызывать из интерфейса)"""
        if mmsi in self.vessels:
            if lat is not None:
                self.vessels[mmsi]['lat'] = lat
            if lon is not None:
                self.vessels[mmsi]['lon'] = lon
            if sog is not None:
                self.vessels[mmsi]['sog'] = sog
            if cog is not None:
                self.vessels[mmsi]['cog'] = cog
            print(f"✏️ Судно {mmsi} обновлено: lat={lat}, lon={lon}, sog={sog}, cog={cog}")

    def get_vessels_info(self) -> Dict:
        """Получить информацию о всех судах"""
        return self.vessels

    def stop_simulation(self):
        """Остановка симуляции"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_next_packet(self) -> Optional[Dict]:
        """Получение следующего пакета"""
        try:
            return self.queue.get_nowait()
        except:
            return None