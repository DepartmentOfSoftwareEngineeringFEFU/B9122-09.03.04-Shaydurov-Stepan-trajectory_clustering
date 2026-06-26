"""
AIS Worker — отдельный процесс с защитой от дублей по каждому MMSI
"""
import asyncio
import json
import sys
import websockets
import time

# Получаем API ключ из аргументов командной строки
if len(sys.argv) > 1:
    API_KEY = sys.argv[1]  # запасной вариант

if len(sys.argv) > 2:
    REGION = sys.argv[2]
else:
    REGION = "far_east"

# Словарь для хранения последнего времени получения для каждого MMSI
last_packet_time = {}
MIN_TIME_BETWEEN_SAME_MMSI = 60  # 60 секунд между одинаковыми MMSI

# Настройки границ для разных регионов
BOUNDING_BOXES = {
    "far_east": [[[40.0, 110.0], [70.0, 190.0]]],
    "vladivostok": [[[42.5, 131.5], [43.5, 132.5]]],
    "global": [[[-90, -180], [90, 180]]]
}

async def main():
    while True:
        try:
            print("DEBUG: Connecting...", file=sys.stderr)
            async with websockets.connect("wss://stream.aisstream.io/v0/stream") as websocket:
                print("DEBUG: Connected!", file=sys.stderr)

                subscribe = {
                    "APIKey": API_KEY,
                    "BoundingBoxes": BOUNDING_BOXES.get(REGION, BOUNDING_BOXES["far_east"]),
                    "FilterMessageTypes": ["PositionReport"]
                }
                await websocket.send(json.dumps(subscribe))
                print(f"DEBUG: Subscribed to {REGION}", file=sys.stderr)

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        if data.get('MessageType') == 'PositionReport':
                            report = data.get('Message', {}).get('PositionReport', {})

                            mmsi = report.get('UserID', 0)
                            lat = report.get('Latitude', 0)
                            lon = report.get('Longitude', 0)
                            current_time = time.time()

                            # Проверяем, не отправляли ли мы пакет с этим MMSI менее 60 секунд назад
                            if mmsi in last_packet_time:
                                elapsed = current_time - last_packet_time[mmsi]
                                if elapsed < MIN_TIME_BETWEEN_SAME_MMSI:
                                    # Слишком часто — пропускаем
                                    continue

                            # Обновляем время последнего пакета для этого MMSI
                            last_packet_time[mmsi] = current_time

                            packet = {
                                'mmsi': mmsi,
                                'lat': lat,
                                'lon': lon,
                                'sog': report.get('SOG', 0),
                                'cog': report.get('COG', 0),
                                'type': 'Судно',
                                'source': 'REAL_AIS'
                            }
                            print(json.dumps(packet))
                            sys.stdout.flush()

                            # Очищаем старые записи (старше 10 минут)
                            for old_mmsi in list(last_packet_time.keys()):
                                if current_time - last_packet_time[old_mmsi] > 600:
                                    del last_packet_time[old_mmsi]

                    except Exception as e:
                        print(f"ERROR: {e}", file=sys.stderr)

        except Exception as e:
            print(f"ERROR: {e}, переподключение через 5 сек...", file=sys.stderr)
            await asyncio.sleep(5)

if __name__ == "__main__":
    asyncio.run(main())