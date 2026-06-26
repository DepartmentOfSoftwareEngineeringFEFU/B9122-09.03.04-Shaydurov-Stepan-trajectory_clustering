"""
Модуль для работы с реальным AIS WebSocket
"""

import subprocess
import threading
import json
import sys
import os
import pandas as pd
from typing import Dict, Optional, Callable
from queue import Queue


class AISRealTimeStream:
    def __init__(self):
        self.process = None
        self.is_connected = False
        self.running = False
        self.thread = None
        self.queue = Queue()

    def connect(self, api_key: str, on_message_callback: Callable = None, region: str = "far_east"):
        """
        Подключение к AIS WebSocket
        Parameters:
        api_key : str
            API ключ для AISstream.io (обязательный параметр)
        on_message_callback : Callable
            Функция обратного вызова для обработки сообщений
        region : str
            Регион подписки ('far_east', 'vladivostok', 'global')
        """
        self.running = True

        if not api_key:
            print("❌ API ключ не указан!")
            return

        # Путь к worker
        worker_path = os.path.join(os.path.dirname(__file__), "ais_worker.py")

        def read_output():
            try:
                # Передаём API ключ в worker как аргумент командной строки
                print(f"🚀 Запуск: {sys.executable} {worker_path} {api_key}")
                self.process = subprocess.Popen(
                    [sys.executable, worker_path, api_key, region],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    bufsize=1
                )
                self.is_connected = True
                print("✅ AIS процесс запущен")

                # Читаем stderr для отладки
                def read_stderr():
                    for line in self.process.stderr:
                        if line.strip():
                            print(f"[WORKER] {line.strip()}")

                stderr_thread = threading.Thread(target=read_stderr, daemon=True)
                stderr_thread.start()

                # Читаем stdout
                for line in self.process.stdout:
                    if not self.running:
                        break
                    line = line.strip()
                    if line:
                        try:
                            packet = json.loads(line)
                            packet['timestamp'] = pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3]
                            self.queue.put(packet)
                            print(f"📥 Пакет в очередь: MMSI={packet['mmsi']}")
                        except Exception as e:
                            print(f"⚠️ Ошибка JSON: {e} | Строка: {line[:100]}")

            except Exception as e:
                print(f"❌ Ошибка: {e}")
            finally:
                self.is_connected = False
                print("🔌 AIS процесс остановлен")

        self.thread = threading.Thread(target=read_output, daemon=True)
        self.thread.start()

    def get_next_packet(self):
        try:
            packet = self.queue.get_nowait()
            return packet
        except:
            return None

    def disconnect(self):
        print("🔌 Остановка AIS процесса...")
        self.running = False
        if self.process:
            self.process.terminate()
        self.is_connected = False

    def is_alive(self) -> bool:
        return self.is_connected and self.process and self.process.poll() is None


class AISMockStream:
    def __init__(self, df):
        self.df = df
        self.is_connected = False
        self.running = False
        self.sim_thread = None
        self.queue = Queue()

    def connect(self, on_message_callback: Callable = None, region: str = "far_east"):
        self.is_connected = True
        self.running = True

        def simulate():
            import random
            import time
            vessels = self.df['mmsi'].unique().tolist()

            while self.running:
                vessel_mmsi = random.choice(vessels)
                vessel_data = self.df[self.df['mmsi'] == vessel_mmsi].iloc[0]

                packet = {
                    'timestamp': pd.Timestamp.now().strftime('%H:%M:%S.%f')[:-3],
                    'mmsi': vessel_mmsi,
                    'lat': vessel_data['lat'] + random.uniform(-0.01, 0.01),
                    'lon': vessel_data['lon'] + random.uniform(-0.01, 0.01),
                    'sog': max(0, vessel_data['sog'] + random.uniform(-2, 2)),
                    'cog': (vessel_data['cog'] + random.uniform(-10, 10)) % 360,
                    'type': vessel_data.get('type', 'Судно'),
                    'source': 'SIMULATION'
                }
                self.queue.put(packet)
                time.sleep(2)

        self.sim_thread = threading.Thread(target=simulate, daemon=True)
        self.sim_thread.start()

    def get_next_packet(self):
        try:
            return self.queue.get_nowait()
        except:
            return None

    def disconnect(self):
        self.running = False

    def is_alive(self) -> bool:
        return self.is_connected