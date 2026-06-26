"""
Модуль загрузки и предобработки траекторных данных
Поддерживает загрузку из CSV, обработку AIS данных, фильтрацию
"""

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from typing import Optional

warnings.filterwarnings('ignore')


class DataLoader:
    """Класс для загрузки и предобработки траекторных данных"""

    def __init__(self):
        self.df = None
        self.original_columns = None

    def validate_dataframe(self, df: pd.DataFrame, source_name: str = "данных") -> tuple[bool, str, pd.DataFrame]:
        if df is None or df.empty:
            return False, f"❌ {source_name}: файл пуст или не содержит данных", None

        df_clean = df.copy()

        # ========== 1. ПРИВЕДЕНИЕ НАЗВАНИЙ КОЛОНОК ==========
        column_mapping = {
            'mmsi': ['mmsi', 'MMSI', 'userid', 'UserID', 'ship_id', 'id', 'vessel_id'],
            'lat': ['lat', 'latitude', 'Latitude', 'LAT', 'y'],
            'lon': ['lon', 'longitude', 'Longitude', 'LON', 'x'],
            'timestamp': ['timestamp', 'time', 'datetime', 'DateTime', 'TIMESTAMP', 't', 'date_time', 'tstamp'],
            'sog': ['sog', 'SOG', 'speed', 'Speed', 'velocity', 'spd'],
            'cog': ['cog', 'COG', 'course', 'Course', 'heading', 'Heading']
        }

        for standard_name, possible_names in column_mapping.items():
            for possible_name in possible_names:
                if possible_name in df_clean.columns and standard_name not in df_clean.columns:
                    df_clean = df_clean.rename(columns={possible_name: standard_name})
                    print(f"  Переименована колонка: {possible_name} -> {standard_name}")
                    break

        # ========== 2. ПРОВЕРКА ОБЯЗАТЕЛЬНЫХ КОЛОНОК ==========
        required_columns = ['mmsi', 'lat', 'lon', 'timestamp', 'sog', 'cog']
        missing_columns = [col for col in required_columns if col not in df_clean.columns]

        if missing_columns:
            error_msg = f"❌ Отсутствуют обязательные колонки: {', '.join(missing_columns)}\n"
            error_msg += f"   Доступные колонки: {list(df_clean.columns)}"
            return False, error_msg, None

        # ========== 3. ПРОВЕРКА ТИПОВ ДАННЫХ ==========
        # MMSI
        try:
            df_clean['mmsi'] = df_clean['mmsi'].astype(str).str.extract('(\d+)')[0]
            df_clean['mmsi'] = df_clean['mmsi'].fillna('0').astype(int)
        except:
            return False, "❌ Не удалось преобразовать MMSI в корректный формат", None

        # Координаты
        try:
            df_clean['lat'] = pd.to_numeric(df_clean['lat'], errors='coerce')
            invalid_lat = df_clean['lat'].isna().sum()
            if invalid_lat > 0:
                df_clean = df_clean[df_clean['lat'].notna()]
                print(f"  ⚠️ Удалено {invalid_lat} строк с некорректной широтой")
        except:
            return False, "❌ Некорректный формат широты (lat)", None

        try:
            df_clean['lon'] = pd.to_numeric(df_clean['lon'], errors='coerce')
            invalid_lon = df_clean['lon'].isna().sum()
            if invalid_lon > 0:
                df_clean = df_clean[df_clean['lon'].notna()]
                print(f"  ⚠️ Удалено {invalid_lon} строк с некорректной долготой")
        except:
            return False, "❌ Некорректный формат долготы (lon)", None

        # Скорость
        try:
            df_clean['sog'] = pd.to_numeric(df_clean['sog'], errors='coerce').fillna(0)
            df_clean['sog'] = df_clean['sog'].clip(0, 102.2)
        except:
            return False, "❌ Некорректный формат скорости (sog)", None

        # Курс
        try:
            df_clean['cog'] = pd.to_numeric(df_clean['cog'], errors='coerce').fillna(0)
            df_clean['cog'] = df_clean['cog'] % 360
        except:
            return False, "❌ Некорректный формат курса (cog)", None

        # ========== TIMESTAMP  ==========
        try:
            # Преобразуем в datetime
            df_clean['timestamp'] = pd.to_datetime(df_clean['timestamp'], errors='coerce')

            invalid_ts = df_clean['timestamp'].isna().sum()
            if invalid_ts > 0:
                print(
                    f"  ⚠️ Пример некорректного timestamp: {df_clean[df_clean['timestamp'].isna()]['timestamp'].iloc[0] if invalid_ts > 0 else 'N/A'}")
                df_clean = df_clean[df_clean['timestamp'].notna()]
                print(f"  ⚠️ Удалено {invalid_ts} строк с некорректной временной меткой")

            if df_clean.empty:
                return False, "❌ Некорректный формат временной метки (timestamp)", None

        except Exception as e:
            return False, f"❌ Ошибка преобразования timestamp: {e}", None

        # ========== 4. ПРОВЕРКА ДИАПАЗОНОВ ==========
        out_of_range_lat = ((df_clean['lat'] < -90) | (df_clean['lat'] > 90)).sum()
        if out_of_range_lat > 0:
            df_clean = df_clean[(df_clean['lat'] >= -90) & (df_clean['lat'] <= 90)]
            print(f"  ⚠️ Удалено {out_of_range_lat} строк с широтой вне диапазона")

        out_of_range_lon = ((df_clean['lon'] < -180) | (df_clean['lon'] > 180)).sum()
        if out_of_range_lon > 0:
            df_clean = df_clean[(df_clean['lon'] >= -180) & (df_clean['lon'] <= 180)]
            print(f"  ⚠️ Удалено {out_of_range_lon} строк с долготой вне диапазона")

        out_of_range_sog = ((df_clean['sog'] < 0) | (df_clean['sog'] > 102.2)).sum()
        if out_of_range_sog > 0:
            df_clean['sog'] = df_clean['sog'].clip(0, 102.2)
            print(f"  ⚠️ Скорректировано {out_of_range_sog} значений скорости")

        # ========== 5. СТАТИСТИКА ==========
        unique_mmsi = df_clean['mmsi'].nunique()
        time_min = df_clean['timestamp'].min()
        time_max = df_clean['timestamp'].max()
        time_span = time_max - time_min

        success_msg = (
            f"✅ Валидация {source_name} успешно завершена!\n"
            f"   📊 Записей: {len(df_clean):,}\n"
            f"   🚢 Судов: {unique_mmsi}\n"
            f"   📅 Период: {time_min.strftime('%Y-%m-%d %H:%M')} - {time_max.strftime('%Y-%m-%d %H:%M')}\n"
            f"   ⏱️ Длительность: {time_span}\n"
            f"   📍 Диапазон широт: {df_clean['lat'].min():.2f}° - {df_clean['lat'].max():.2f}°\n"
            f"   📍 Диапазон долгот: {df_clean['lon'].min():.2f}° - {df_clean['lon'].max():.2f}°\n"
            f"   ⚡ Средняя скорость: {df_clean['sog'].mean():.1f} уз"
        )

        return True, success_msg, df_clean


    def load_from_csv(self, file_path, **kwargs):
        try:
            nrows = kwargs.get('nrows', None)
            # Прямое чтение с табуляцией
            df = pd.read_csv(file_path, sep='\t', encoding='utf-8', nrows=nrows)
            print(f"✅ CSV загружен с разделителем: TAB")
            print(f"📊 Найдены колонки: {list(df.columns)}")

            # Очищаем названия колонок от кавычек
            df.columns = df.columns.str.replace('"', '').str.strip()
            print(f"📊 Колонки после очистки: {list(df.columns)}")

            # Если колонок меньше 3, значит разделитель не тот
            if len(df.columns) < 3:
                # Пробуем с точкой с запятой
                df = pd.read_csv(file_path, sep=';', encoding='utf-8', **kwargs)
                df.columns = df.columns.str.replace('"', '').str.strip()
                print(f"📊 Попытка с ';' : {list(df.columns)}")

            if len(df.columns) < 3:
                print("❌ Не удалось правильно прочитать CSV файл")
                return None

            # ========== ПРЕОБРАЗОВАНИЕ ДАННЫХ ==========

            # 1. Заменяем запятую на точку в координатах
            if 'lat' in df.columns and df['lat'].dtype == 'object':
                df['lat'] = df['lat'].astype(str).str.replace(',', '.').astype(float)
            if 'lon' in df.columns and df['lon'].dtype == 'object':
                df['lon'] = df['lon'].astype(str).str.replace(',', '.').astype(float)

            # 2. СОЗДАЁМ timestamp ИЗ date_add
            if 'date_add' in df.columns:
                df['timestamp'] = pd.to_datetime(df['date_add'], format='%d.%m.%Y %H:%M', errors='coerce')
                print(f"📅 Успешно преобразовано timestamp: {df['timestamp'].notna().sum()} из {len(df)} строк")

                if df['timestamp'].notna().sum() == 0:
                    df['timestamp'] = pd.to_datetime(df['date_add'], errors='coerce')
                    print(f"📅 После автоопределения: {df['timestamp'].notna().sum()} из {len(df)} строк")
            else:
                print("⚠️ Колонка 'date_add' не найдена")
                df['timestamp'] = pd.Timestamp.now()

            # 3. Создаём mmsi
            if 'mmsi' not in df.columns:
                if 'id_marine' in df.columns:
                    df['mmsi'] = df['id_marine']
                elif 'id_track' in df.columns:
                    df['mmsi'] = df['id_track']
                else:
                    df['mmsi'] = range(1, len(df) + 1)

            # 4. Создаём sog
            if 'sog' not in df.columns and 'speed' in df.columns:
                df['sog'] = pd.to_numeric(df['speed'], errors='coerce').fillna(0)
            elif 'sog' not in df.columns:
                df['sog'] = 0

            # 5. Создаём cog
            if 'cog' not in df.columns and 'course' in df.columns:
                df['cog'] = pd.to_numeric(df['course'], errors='coerce').fillna(0)
            elif 'cog' not in df.columns:
                df['cog'] = 0

            # 6. Создаём type
            if 'type' not in df.columns:
                df['type'] = 'Судно'

            # 7. Добавляем heading
            if 'heading' not in df.columns:
                df['heading'] = df['cog']

            # Удаляем строки с некорректным timestamp
            before = len(df)
            df = df[df['timestamp'].notna()]
            if before - len(df) > 0:
                print(f"📅 Удалено {before - len(df)} строк с некорректным timestamp")

            if df.empty:
                print("❌ Нет данных с корректным timestamp")
                return None

            self.original_columns = df.columns.tolist()

            # ВАЛИДАЦИЯ
            is_valid, message, validated_df = self.validate_dataframe(df, "CSV файла")

            if not is_valid:
                print(message)
                return None

            print(message)
            self.df = validated_df
            return self.df

        except Exception as e:
            print(f"Ошибка загрузки CSV файла: {e}")
            import traceback
            traceback.print_exc()
            return None

    def load_sample_data(self):
        """
        Создание примера данных для тестирования
        """
        np.random.seed(42)

        trajectories = []

        # Траектория 1: Японское море (центр, вдали от берегов)
        # Координаты: между 133-137° в.д. (центр моря, подальше от берегов)
        traj1_lat = np.linspace(40.0, 43.0, 50) + np.random.normal(0, 0.08, 50)
        traj1_lon = np.linspace(134.5, 136.5, 100) + np.random.normal(0, 0.08, 100)
        trajectories.append((traj1_lat, traj1_lon, 'Грузовое', np.random.uniform(12, 18, 100)))

        # Траектория 2: Охотское море (центр, вдали от берегов)
        # Координаты: между 145-152° в.д.
        traj2_lat = np.linspace(48.0, 56.0, 100) + np.random.normal(0, 0.08, 100)
        traj2_lon = np.linspace(147.0, 150.0, 100) + np.random.normal(0, 0.08, 100)
        trajectories.append((traj2_lat, traj2_lon, 'Рыболовное', np.random.uniform(8, 14, 100)))

        # Траектория 3: Тихий океан (восточнее Курильских островов)
        traj3_lat = np.linspace(44.0, 52.0, 100) + np.random.normal(0, 0.08, 100)
        traj3_lon = np.linspace(155.0, 160.0, 100) + np.random.normal(0, 0.08, 100)
        trajectories.append((traj3_lat, traj3_lon, 'Танкер', np.random.uniform(13, 20, 100)))

        # Траектория 5: Татарский пролив (узкая полоса воды между Сахалином и материком)
        traj5_lat = np.linspace(45.6, 51.5, 80) + np.random.normal(0, 0.05, 80)
        traj5_lon = np.linspace(141.2, 141.8, 80) + np.random.normal(0, 0.03, 80)
        trajectories.append((traj5_lat, traj5_lon, 'Грузовое', np.random.uniform(10, 15, 80)))

        # Траектория 6: Японское море (южная часть, между Кореей и Японией)
        traj6_lat = np.linspace(36.0, 42.0, 90) + np.random.normal(0, 0.08, 90)
        traj6_lon = np.linspace(132.0, 135.0, 90) + np.random.normal(0, 0.08, 90)
        trajectories.append((traj6_lat, traj6_lon, 'Контейнеровоз', np.random.uniform(14, 22, 90)))

        # Траектория 7: Охотское море (южная часть, севернее Хоккайдо)
        traj7_lat = np.linspace(45.0, 49.0, 80) + np.random.normal(0, 0.08, 80)
        traj7_lon = np.linspace(144.0, 148.0, 80) + np.random.normal(0, 0.08, 80)
        trajectories.append((traj7_lat, traj7_lon, 'Рыболовное', np.random.uniform(7, 12, 80)))

        # Сбор всех данных
        all_lats = []
        all_lons = []
        all_types = []
        all_sogs = []
        all_mmsi = []
        all_timestamps = []

        timestamp_start = pd.Timestamp('2026-05-23 00:00:00')

        for i, (lats, lons, ship_type, sogs) in enumerate(trajectories):
            mmsi = f'1234567{i + 1:02d}'
            for j, (lat, lon, sog) in enumerate(zip(lats, lons, sogs)):
                all_lats.append(lat)
                all_lons.append(lon)
                all_types.append(ship_type)
                all_sogs.append(sog)
                all_mmsi.append(mmsi)
                all_timestamps.append(timestamp_start + pd.Timedelta(minutes=j + i * 120))

        # Создание DataFrame
        self.df = pd.DataFrame({
            'mmsi': all_mmsi,
            'timestamp': all_timestamps,
            'lat': all_lats,
            'lon': all_lons,
            'sog': all_sogs,
            'cog': np.random.uniform(0, 360, len(all_lats)),
            'heading': np.random.uniform(0, 360, len(all_lats)),
            'type': all_types
        })

        print(f"Создано {len(self.df)} записей тестовых данных")
        print(f"  - {self.df['mmsi'].nunique()} уникальных судов")
        print(f"  - Типы судов: {self.df['type'].unique().tolist()}")
        print(f"  - Диапазон широт: {self.df['lat'].min():.1f}° - {self.df['lat'].max():.1f}°")
        print(f"  - Диапазон долгот: {self.df['lon'].min():.1f}° - {self.df['lon'].max():.1f}°")

        return self.df

    def clean_data(self, remove_outliers=True, speed_limits=(0, 70), interpolate_gaps=True, max_gap_minutes=10):
        """
        Очистка данных: удаление дубликатов, выбросов, интерполяция пропусков
        """
        if self.df is None or self.df.empty:
            print("⚠️ Нет данных для очистки")
            return self.df  # Возвращаем None, а не self.df

        df_clean = self.df.copy()
        initial_count = len(df_clean)

        # 1. Удаление дубликатов
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)

        # 2. Обработка пропусков
        if 'lat' in df_clean.columns:
            df_clean['lat'] = df_clean['lat'].fillna(df_clean['lat'].median())
        if 'lon' in df_clean.columns:
            df_clean['lon'] = df_clean['lon'].fillna(df_clean['lon'].median())
        if 'sog' in df_clean.columns:
            df_clean['sog'] = df_clean['sog'].fillna(0)
        if 'cog' in df_clean.columns:
            df_clean['cog'] = df_clean['cog'].fillna(0)

        # 3. Удаление выбросов по скорости
        if remove_outliers and 'sog' in df_clean.columns:
            before = len(df_clean)
            df_clean = df_clean[
                (df_clean['sog'] >= speed_limits[0]) &
                (df_clean['sog'] <= speed_limits[1])
                ]
            outliers_removed = before - len(df_clean)
        else:
            outliers_removed = 0

        # 4. Удаление строк с NaN после обработки
        df_clean = df_clean.dropna()

        # 5. Интерполяция пропущенных позиций
        interp_added = 0
        if interpolate_gaps and len(df_clean) > 1:
            try:
                before_interp = len(df_clean)
                df_clean = self.interpolate_trajectories(df_clean, max_gap_minutes)
                if df_clean is not None:
                    interp_added = len(df_clean) - before_interp
                else:
                    interp_added = 0
            except Exception as e:
                print(f"⚠️ Ошибка интерполяции: {e}")

        print(f"Очистка данных завершена:")
        print(f"  - Исходно: {initial_count} записей")
        print(f"  - Удалено дубликатов: {duplicates_removed}")
        print(f"  - Удалено выбросов: {outliers_removed}")
        print(f"  - Добавлено интерполяцией: {interp_added}")
        print(f"  - Итоговое количество: {len(df_clean)}")

        self.df = df_clean
        return df_clean

    def filter_by_time(self, date_from=None, date_to=None):

        if self.df is None or 'timestamp' not in self.df.columns:
            return None

        df_filtered = self.df.copy()

        # Преобразование timestamp в datetime если нужно
        if not pd.api.types.is_datetime64_any_dtype(df_filtered['timestamp']):
            df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

        if date_from:
            df_filtered = df_filtered[df_filtered['timestamp'] >= pd.to_datetime(date_from)]
        if date_to:
            df_filtered = df_filtered[df_filtered['timestamp'] <= pd.to_datetime(date_to)]

        print(f"Фильтрация по времени: {len(df_filtered)} записей")
        return df_filtered

    def filter_by_area(self, lat_min=None, lat_max=None, lon_min=None, lon_max=None):

        if self.df is None:
            return None

        df_filtered = self.df.copy()

        if lat_min is not None:
            df_filtered = df_filtered[df_filtered['lat'] >= lat_min]
        if lat_max is not None:
            df_filtered = df_filtered[df_filtered['lat'] <= lat_max]
        if lon_min is not None:
            df_filtered = df_filtered[df_filtered['lon'] >= lon_min]
        if lon_max is not None:
            df_filtered = df_filtered[df_filtered['lon'] <= lon_max]

        print(f"Фильтрация по области: {len(df_filtered)} записей")
        return df_filtered

    def filter_by_ship_type(self, ship_types):
        if self.df is None or 'type' not in self.df.columns:
            return self.df

        if isinstance(ship_types, str):
            ship_types = [ship_types]

        df_filtered = self.df[self.df['type'].isin(ship_types)]
        print(f"Фильтрация по типу судна: {len(df_filtered)} записей")
        return df_filtered

    def get_data_summary(self):

        if self.df is None:
            return {"error": "Данные не загружены"}

        summary = {
            "total_records": len(self.df),
            "columns": list(self.df.columns),
            "date_range": None,
            "lat_range": None,
            "lon_range": None,
            "unique_ships": None,
            "missing_values": self.df.isnull().sum().to_dict()
        }

        if 'timestamp' in self.df.columns:
            if pd.api.types.is_datetime64_any_dtype(self.df['timestamp']):
                summary["date_range"] = {
                    "from": self.df['timestamp'].min(),
                    "to": self.df['timestamp'].max()
                }

        if 'lat' in self.df.columns:
            summary["lat_range"] = (self.df['lat'].min(), self.df['lat'].max())

        if 'lon' in self.df.columns:
            summary["lon_range"] = (self.df['lon'].min(), self.df['lon'].max())

        if 'mmsi' in self.df.columns:
            summary["unique_ships"] = self.df['mmsi'].nunique()

        return summary

    def interpolate_trajectories(self, df: pd.DataFrame, max_gap_minutes: int = 10) -> pd.DataFrame:
        """
        Интерполяция пропущенных позиций для построения непрерывных траекторий
        """
        if df is None or df.empty:
            return df

        # Проверяем наличие необходимых колонок для интерполяции
        required_cols = ['mmsi', 'timestamp', 'lat', 'lon']
        missing_cols = [col for col in required_cols if col not in df.columns]

        # Если нет нужных колонок, возвращаем исходные данные без изменений
        if missing_cols:
            print(f"⚠️ Отсутствуют колонки для интерполяции: {missing_cols}")
            return df

        # ========== 1.  ИСХОДНЫЕ ДАННЫЕ ==========
        # Помечаем исходные точки
        df_copy = df.copy()
        df_copy['_original'] = True

        if not pd.api.types.is_datetime64_any_dtype(df_copy['timestamp']):
            df_copy['timestamp'] = pd.to_datetime(df_copy['timestamp'], errors='coerce')


        valid_ts_mask = df_copy['timestamp'].notna()

        if not valid_ts_mask.any():
            print("⚠️ Нет корректных временных меток для интерполяции")
            return df

        # Данные с корректным timestamp (для интерполяции)
        df_valid = df_copy[valid_ts_mask].copy()
        # Данные с некорректным timestamp (просто вернём их как есть)
        df_invalid = df_copy[~valid_ts_mask].copy()

        # Сортируем по MMSI и времени
        df_valid = df_valid.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)

        interpolated_rows = []

        # Группируем по каждому судну
        for mmsi, group in df_valid.groupby('mmsi'):
            if len(group) < 2:
                continue

            group = group.reset_index(drop=True)

            # Проходим по точкам и ищем пропуски
            for i in range(len(group) - 1):
                prev_row = group.iloc[i]
                curr_row = group.iloc[i + 1]

                #  разница в минутах
                time_diff = (curr_row['timestamp'] - prev_row['timestamp']).total_seconds() / 60

                #  разрыв между 1 и max_gap_minutes минутами
                if 1 <= time_diff <= max_gap_minutes:
                    num_points = int(time_diff)

                # для каждой промежуточной минуты
                    for j in range(1, num_points):
                        ratio = j / num_points

                            # линейная интерполяция координат
                        interp_lat = prev_row['lat'] + (curr_row['lat'] - prev_row['lat']) * ratio
                        interp_lon = prev_row['lon'] + (curr_row['lon'] - prev_row['lon']) * ratio

                        # Интерполяция скорости
                        if 'sog' in df.columns:
                            interp_sog = prev_row.get('sog', 0) + (
                                        curr_row.get('sog', 0) - prev_row.get('sog', 0)) * ratio
                            interp_sog = round(max(0, interp_sog), 1)
                        else:
                            interp_sog = 0

                        # Интерполяция курса
                        if 'cog' in df.columns:
                            interp_cog = prev_row.get('cog', 0) + (
                                        curr_row.get('cog', 0) - prev_row.get('cog', 0)) * ratio
                            interp_cog = interp_cog % 360
                            interp_cog = round(interp_cog, 1)
                        else:
                            interp_cog = 0

                        # Интерполированное время
                        interp_time = prev_row['timestamp'] + pd.Timedelta(minutes=j)

                        # Формируем интерполированную точку
                        new_point = {
                            'mmsi': mmsi,
                            'timestamp': interp_time,
                            'lat': round(interp_lat, 6),
                            'lon': round(interp_lon, 6),
                            'sog': interp_sog,
                            'cog': interp_cog,
                            '_original': False,  # Помечаем как интерполированную
                            '_interpolated': True
                        }

                        # Копируем остальные колонки из предыдущей точки
                        for col in df.columns:
                            if col not in new_point:
                                new_point[col] = prev_row.get(col, None)

                        interpolated_rows.append(new_point)

        # ========== 2. ОБЪЕДИНЯЕМ ВСЕ ДАННЫЕ ==========
        result_parts = []

        # Добавляем исходные данные (с корректным timestamp)
        result_parts.append(df_valid)

        # Добавляем интерполированные точки
        if interpolated_rows:
            interp_df = pd.DataFrame(interpolated_rows)
            result_parts.append(interp_df)
            print(f"✅ Добавлено {len(interpolated_rows)} интерполированных точек")

        # Добавляем данные с некорректным timestamp (возвращаем как есть)
        if not df_invalid.empty:
            result_parts.append(df_invalid)
            print(f"📁 Сохранено {len(df_invalid)} строк с некорректным timestamp")

        # Объединяем всё
        if len(result_parts) > 1:
            df_result = pd.concat(result_parts, ignore_index=True)
        else:
            df_result = df_valid

        # Удаляем временные колонки
        if '_original' in df_result.columns:
            df_result = df_result.drop(columns=['_original'])

        # Сортируем по MMSI и времени
        if not df_result.empty:
            df_result = df_result.sort_values(['mmsi', 'timestamp']).reset_index(drop=True)

        print(
            f"📊 Итоговое количество записей: {len(df_result)} (исходные: {len(df_valid) + len(df_invalid)}, интерполированные: {len(interpolated_rows)})")

        return df_result
