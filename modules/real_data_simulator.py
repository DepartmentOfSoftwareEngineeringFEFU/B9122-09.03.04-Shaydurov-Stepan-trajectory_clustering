"""
Модуль симуляции реальных траекторных данных
Воспроизводит исторические AIS записи в реальном времени
"""

import pandas as pd
import threading
import time
import os
import random
from queue import Queue
from typing import Dict, Optional


class RealDataSimulator:
    """Симулятор, воспроизводящий реальные траекторные данные из CSV"""

    def __init__(self):
        self.running = False
        self.thread = None
        self.queue = None
        self.trajectories = {}
        self.active_tracks = []
        self.update_interval = 2.0
        self.max_rows = None

    def set_max_rows(self, max_rows: int = None):
        """Установка ограничения на количество загружаемых строк"""
        self.max_rows = max_rows

    def set_queue(self, queue):
        """Устанавливает внешнюю очередь"""
        self.queue = queue

    def display_all_points(self):
        """Отправляет ВСЕ точки в очередь сразу (без симуляции)"""
        if self.queue is None:
            return 0

        points_sent = 0
        for track_id, track in self.trajectories.items():
            for point in track['points']:
                packet = {
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3],
                    'mmsi': track['mmsi'],
                    'lat': point['lat'],
                    'lon': point['lon'],
                    'sog': point.get('sog', 0),
                    'cog': point.get('cog', 0),
                    'type': 'Реальное судно',
                    'source': 'REAL_DATA',
                    'track_id': track_id,
                }
                self.queue.put(packet)
                points_sent += 1
        return points_sent

    def load_data_from_csv(self, csv_path: str, max_rows: int = None) -> bool:
        """Загрузка данных из CSV файла"""
        try:
            limit = max_rows or self.max_rows

            # Определяем разделитель
            for sep in [',', ';', '\t']:
                try:
                    if limit:
                        df = pd.read_csv(csv_path, sep=sep, encoding='utf-8', nrows=limit)
                    else:
                        df = pd.read_csv(csv_path, sep=sep, encoding='utf-8')
                    if 'lat' in df.columns and 'lon' in df.columns:
                        break
                except:
                    continue
            else:
                if limit:
                    df = pd.read_csv(csv_path, encoding='utf-8', nrows=limit)
                else:
                    df = pd.read_csv(csv_path, encoding='utf-8')

            if df.empty:
                return False

            # Заменяем запятую на точку в координатах
            if df['lat'].dtype == 'object':
                df['lat'] = df['lat'].astype(str).str.replace(',', '.').astype(float)
            if df['lon'].dtype == 'object':
                df['lon'] = df['lon'].astype(str).str.replace(',', '.').astype(float)

            # Создаём идентификатор трека
            if 'id_track' not in df.columns:
                if 'id_marine' in df.columns:
                    df['id_track'] = df['id_marine']
                else:
                    df['id_track'] = range(len(df))

            # Добавляем скорость и курс
            if 'sog' not in df.columns and 'speed' in df.columns:
                df['sog'] = df['speed']
            else:
                df['sog'] = 0

            if 'cog' not in df.columns and 'course' in df.columns:
                df['cog'] = df['course']
            else:
                df['cog'] = 0

            # Группируем по id_track
            self.trajectories = {}
            for track_id, group in df.groupby('id_track'):
                points = []
                for _, row in group.iterrows():
                    points.append({
                        'lat': row['lat'],
                        'lon': row['lon'],
                        'sog': row.get('sog', 0),
                        'cog': row.get('cog', 0),
                    })

                if points:
                    self.trajectories[track_id] = {
                        'points': points,
                        'current_index': 0,
                        'last_time': None,
                        'mmsi': 273210000 + (int(track_id) % 100000) if isinstance(track_id, (int, float)) else 273210000 + (hash(str(track_id)) % 100000)
                    }

            return len(self.trajectories) > 0

        except Exception as e:
            print(f"Ошибка загрузки данных: {e}")
            return False

    def load_directory(self, directory_path: str, max_rows_per_file: int = None) -> bool:
        """Загрузка всех CSV файлов из директории"""
        self.data_dir = directory_path
        total_tracks = 0

        for root, dirs, files in os.walk(directory_path):
            for file in files:
                if file.endswith('.csv'):
                    csv_path = os.path.join(root, file)
                    if self.load_data_from_csv(csv_path, max_rows_per_file):
                        total_tracks += len(self.trajectories)

        return total_tracks > 0

    def start_simulation(self, num_tracks: int = None, update_interval: float = 2.0):
        """Запуск симуляции на основе реальных данных"""
        if not self.trajectories:
            return

        self.running = True
        self.update_interval = update_interval

        track_ids = list(self.trajectories.keys())
        if num_tracks is not None and num_tracks < len(track_ids):
            track_ids = random.sample(track_ids, num_tracks)

        self.active_tracks = track_ids.copy()

        for track_id in self.active_tracks:
            self.trajectories[track_id]['current_index'] = 0

        def simulate():
            for track_id in self.active_tracks:
                self._send_point(track_id)

            while self.running:
                time.sleep(update_interval)

                for track_id in self.active_tracks[:]:
                    track = self.trajectories[track_id]
                    if track['current_index'] < len(track['points']) - 1:
                        track['current_index'] += 1
                        self._send_point(track_id)
                    else:
                        self.active_tracks.remove(track_id)

                if len(self.active_tracks) == 0:
                    self.running = False
                    break

        self.thread = threading.Thread(target=simulate, daemon=True)
        self.thread.start()

    def _send_point(self, track_id):
        """Отправка текущей точки трека в очередь"""
        track = self.trajectories[track_id]
        point = track['points'][track['current_index']]

        packet = {
            'timestamp': pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3],
            'mmsi': track['mmsi'],
            'lat': point['lat'],
            'lon': point['lon'],
            'sog': point.get('sog', 0),
            'cog': point.get('cog', 0),
            'type': 'Реальное судно',
            'source': 'REAL_DATA',
            'track_id': track_id,
        }

        self.queue.put(packet)

    def stop_simulation(self):
        """Остановка симуляции"""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1)

    def get_next_packet(self) -> Optional[Dict]:
        """Получение следующего пакета из очереди"""
        try:
            return self.queue.get_nowait()
        except:
            return None

    def get_statistics(self) -> Dict:
        """Получение статистики по загруженным данным"""
        total_points = sum(len(t['points']) for t in self.trajectories.values())
        return {
            'total_tracks': len(self.trajectories),
            'total_points': total_points,
            'active_tracks': len(self.active_tracks) if self.running else 0
        }