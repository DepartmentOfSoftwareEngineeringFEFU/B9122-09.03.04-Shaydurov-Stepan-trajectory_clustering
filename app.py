import streamlit as st
import os
import tempfile
import pandas as pd
import time
from datetime import datetime
import numpy as np
from streamlit_folium import folium_static
import folium
from modules.data_loader import DataLoader
from modules.clustering import ClusterAnalyzer
from modules.visualization import MapVisualizer, ChartVisualizer, save_analysis_to_history
from modules.ais_stream import AISRealTimeStream, AISMockStream
from modules.export_utils import export_to_geojson, export_to_csv
import matplotlib.pyplot as plt

# Настройка страницы
st.set_page_config(
    page_title="Анализ траекторных данных",
    page_icon="🚢",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Заголовок приложения
col1, col2, col3 = st.columns([1, 6, 1])
with col2:
    st.title("Анализ траекторных данных")
    st.markdown(
        "<p style='text-align: center; color: #6c7a8a;'>Комплексный кластерный анализ траекторных данных акваторий Дальнего Востока</p>",
        unsafe_allow_html=True)
    st.markdown("---")

# Инициализация состояния сессии
if 'streaming_points' not in st.session_state:
    st.session_state.streaming_points = []
if 'processed_keys' not in st.session_state:
    st.session_state.processed_keys = set()
if 'real_sim_queue' not in st.session_state:
    from queue import Queue
    st.session_state.real_sim_queue = Queue()
if 'data_loaded' not in st.session_state:
    st.session_state.data_loaded = False
if 'clustering_done' not in st.session_state:
    st.session_state.clustering_done = False
if 'df' not in st.session_state:
    st.session_state.df = None
if 'df_original' not in st.session_state:
    st.session_state.df_original = None
if 'cluster_labels' not in st.session_state:
    st.session_state.cluster_labels = None
if 'clustering_metrics' not in st.session_state:
    st.session_state.clustering_metrics = None
if 'current_algorithm' not in st.session_state:
    st.session_state.current_algorithm = 'DBSCAN'
if 'analysis_history' not in st.session_state:
    st.session_state.analysis_history = []
if 'csv_loaded' not in st.session_state:
    st.session_state.csv_loaded = False

@st.cache_resource
def get_loader():
    return DataLoader()


@st.cache_resource
def get_clusterer():
    return ClusterAnalyzer()


loader = get_loader()
clusterer = get_clusterer()


# ==================== БОКОВАЯ ПАНЕЛЬ ====================
with st.sidebar:
    st.markdown("## Загрузка данных")

    data_source = st.radio("Источник данных",
        ["Пример данных", "Загрузить файл (CSV)",
         "Онлайн-имитация", "Aisstream.io (онлайн)"],
        help="Выберите источник данных"
    )

    # ========== ОБРАБОТКА ИСТОЧНИКОВ ДАННЫХ ==========

    if data_source == "Aisstream.io (онлайн)":

        st.markdown("### Подключение к реальному AIS потоку")

        st.info("🔗 Источник: AISstream.io ")

        # Поле для ввода API ключа

        api_key_input = st.text_input(

            "🔑 API ключ AISstream.io",

            type="password",

            placeholder="Введите ваш API ключ",

            key="ais_api_key"

        )

        col1, col2 = st.columns(2)

        with col1:

            auto_refresh_ais = st.checkbox("🔄 Автообновление карты", value=True, key="auto_refresh_ais")

        with col2:

            refresh_interval_ais = st.slider("Интервал обновления (сек)", 1, 10, 3, key="refresh_interval_ais")

        if st.button("🔌 ПОДКЛЮЧИТЬСЯ К AIS", key="ais_connect", use_container_width=True, type="primary"):

            if not api_key_input:

                st.error("❌ Введите API ключ!")

            else:

                from modules.ais_stream import AISRealTimeStream

                if 'online_sim' in st.session_state and st.session_state.online_sim:
                    st.session_state.online_sim.stop_simulation()

                    st.session_state.online_sim = None

                st.session_state.sim_active = False

                st.session_state.ais_stream = AISRealTimeStream()

                st.session_state.ais_stream.connect(api_key_input, None, region_select)

                st.session_state.ais_active = True

                st.session_state.data_loaded = True

                st.session_state.stream_mode = "Aisstream.io"

                st.session_state.streaming_points = []

                st.session_state.processed_keys = set()

                st.success("✅ Подключение к реальному AIS установлено!")

                st.rerun()

        if st.button("🔌 ОТКЛЮЧИТЬСЯ", key="ais_disconnect", use_container_width=True):

            if 'ais_stream' in st.session_state and st.session_state.ais_stream:
                st.session_state.ais_stream.disconnect()

                st.session_state.ais_stream = None

            st.session_state.ais_active = False

            st.session_state.stream_mode = "ОСТАНОВЛЕН"

            st.success("✅ Отключено от AIS")

            st.rerun()

        if st.session_state.get('ais_active', False):

            st.success("🟢 **Статус: ПОДКЛЮЧЕН**")

        else:

            st.warning("⚪ **Статус: ОТКЛЮЧЕН**")

        st.markdown("---")

        st.caption(" Данные с AISstream.io")

    elif data_source == "Загрузить файл (CSV)":

        st.markdown("### Загрузка данных из CSV")

        st.info("📁 Загрузите CSV файл с данными судов")

        max_rows = st.number_input(

            "Максимум строк для загрузки (0 = все)",

            min_value=0, max_value=100000, value=100, step=100,

            key="csv_max_rows"

        )

        # Инициализация

        if 'csv_file_key' not in st.session_state:
            st.session_state.csv_file_key = None

        uploaded_file = st.file_uploader(

            "Выберите CSV файл",

            type=['csv'],

            key="csv_upload"

        )

        if uploaded_file is not None:

            file_key = f"{uploaded_file.name}_{uploaded_file.size}"

            # Загружаем только если файл новый

            if st.session_state.csv_file_key != file_key:

                st.session_state.csv_file_key = file_key

                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8') as f:

                    f.write(uploaded_file.getvalue().decode('utf-8'))

                    temp_path = f.name

                try:

                    if max_rows > 0:

                        df = loader.load_from_csv(temp_path, nrows=max_rows)

                    else:

                        df = loader.load_from_csv(temp_path)

                    if df is not None and len(df) > 0:

                        st.session_state.df = df

                        st.session_state.df_original = df.copy()

                        st.session_state.data_loaded = True

                        st.session_state.is_interpolated = False  #  Новый флаг

                        st.success(f"✅ Загружено {len(df)} записей")

                        st.rerun()

                    else:

                        st.error("❌ Не удалось загрузить данные")


                except Exception as e:

                    st.error(f"❌ Ошибка: {e}")

                finally:

                    os.unlink(temp_path)

        # Показываем текущие данные

        if st.session_state.df is not None and st.session_state.data_loaded:
            st.info(f"📁 Текущие данные: {len(st.session_state.df)} записей")

        # Управление отображением

        if st.session_state.get('real_sim') is not None:

            st.markdown("---")

            st.markdown("###  Управление отображением")

            stats = st.session_state.real_sim.get_statistics()

            st.info(f" Загружено {stats['total_tracks']} судов, {stats['total_points']} точек")

            col_display, col_clear = st.columns(2)

            with col_display:

                if st.button("🗺️ ПОКАЗАТЬ ВСЕ ТОЧКИ НА КАРТЕ", use_container_width=True):
                    st.session_state.streaming_points = []

                    st.session_state.processed_keys = set()

                    points_sent = st.session_state.real_sim.display_all_points()

                    st.success(f"✅ Отправлено {points_sent} точек на карту")

                    st.session_state.sim_active = True

                    st.session_state.stream_mode = "REAL_DATA"

                    st.rerun()

            with col_clear:

                if st.button("🗑️ ОЧИСТИТЬ КАРТУ", use_container_width=True):
                    st.session_state.streaming_points = []

                    st.session_state.processed_keys = set()

                    st.session_state.sim_active = False

                    st.session_state.stream_mode = "REAL_DATA_LOADED"

                    st.success("🗑️ Карта очищена")

                    st.rerun()


    elif data_source == "Онлайн-имитация":

        col_sim1, col_sim2 = st.columns(2)

        auto_refresh = st.checkbox("🔄 АВТООБНОВЛЕНИЕ СТРАНИЦЫ", value=True, key="auto_refresh")

        refresh_interval = st.slider("Интервал обновления (сек)", 1, 10, 3, key="refresh_interval")

        if st.button("🚢 ЗАПУСТИТЬ ИМИТАЦИЮ", key="sim_start", use_container_width=True):

            from modules.online_simulator import OnlineSimulator

            # Очистка

            st.session_state.streaming_points = []

            st.session_state.processed_keys = set()

            # Запускаем симуляцию

            st.session_state.online_sim = OnlineSimulator()

            st.session_state.online_sim.start_simulation(5, 3)

            st.session_state.sim_active = True

            st.session_state.data_loaded = True

            st.session_state.stream_mode = "SIMULATION"



            if st.session_state.df is None:

                # Создаём маленький пустой DF только для структуры

                empty_df = pd.DataFrame(columns=['mmsi', 'lat', 'lon', 'timestamp', 'sog', 'cog', 'type'])

                is_valid, msg, validated_df = loader.validate_dataframe(empty_df, "имитации данных")

                if is_valid:
                    st.session_state.df = validated_df

                    st.session_state.df_original = validated_df.copy()

            st.success(f"✅ Имитация запущена: {5} судов")

            st.rerun()

        if st.button("⏹️ ОСТАНОВИТЬ ИМИТАЦИЮ", key="sim_stop", use_container_width=True):

            if 'online_sim' in st.session_state and st.session_state.online_sim:
                st.session_state.online_sim.stop_simulation()

                st.session_state.online_sim = None

            st.session_state.sim_active = False
            st.session_state.stream_mode = "ОСТАНОВЛЕН"
            st.success("⏹️ Имитация остановлена")
            st.rerun()


    else:  # "Пример данных (тестовые)"
        if st.button("Пример данных", use_container_width=True):
            df = loader.load_sample_data()
            st.session_state.df = df
            st.session_state.df_original = df.copy()
            st.session_state.data_loaded = True
            st.success(f"✅ Загружено {len(df)} записей")

    # ========== ЭТИ БЛОКИ (если данные загружены) ==========
    if st.session_state.data_loaded and st.session_state.df is not None:
        st.markdown("---")

        if st.button("🔄 Сбросить все результаты", use_container_width=True):
            st.session_state.clustering_done = False
            st.session_state.cluster_labels = None
            st.session_state.clustering_metrics = None
            if st.session_state.df is not None and 'cluster' in st.session_state.df.columns:
                st.session_state.df = st.session_state.df.drop(columns=['cluster'])

            # Сброс интерполяции
            if st.session_state.df_original is not None:
                st.session_state.df = st.session_state.df_original.copy()

            if 'online_sim' in st.session_state and st.session_state.online_sim:
                st.session_state.online_sim.stop_simulation()
                st.session_state.online_sim = None
            st.session_state.sim_active = False
            st.session_state.streaming_points = []
            st.session_state.processed_keys = set()
            st.session_state.stream_mode = "ОСТАНОВЛЕН"
            st.session_state.last_csv_file = None

            st.success("✅ Состояние сброшено")
            st.rerun()

        # ==================== ПРЕДОБРАБОТКА ====================
        st.markdown("##  Предобработка данных")

        col_clean1, col_clean2 = st.columns(2)
        with col_clean1:
            if st.button("🗑️ Очистить данные (дубликаты, выбросы)", use_container_width=True):
                if st.session_state.df is not None:
                    loader.clean_data(remove_outliers=True, speed_limits=(0, 70), interpolate_gaps=False)
                    st.session_state.df = loader.df
                    st.success(f"✅ Очистка завершена. Осталось {len(st.session_state.df)} записей")
                    st.rerun()
                else:
                    st.warning("⚠️ Нет данных для очистки")

        with col_clean2:
            if st.button("📈 Интерполяция пропусков", use_container_width=True):
                if st.session_state.df is not None:
                    if st.session_state.get('sim_active', False):
                        st.warning("⚠️ Интерполяция не применяется к данным онлайн-имитации")
                    elif st.session_state.get('ais_active', False):
                        st.warning("⚠️ Интерполяция не применяется к реальному AIS потоку")
                    else:
                        try:
                            old_count = len(st.session_state.df)

                            # Вызываем интерполяцию напрямую
                            result_df = loader.interpolate_trajectories(
                                st.session_state.df.copy(),
                                max_gap_minutes=10
                            )

                            st.session_state.df = result_df
                            st.session_state.is_interpolated = True  # ← Флаг, что данные интерполированы
                            st.session_state.streaming_points = result_df.to_dict('records')
                            st.session_state.processed_keys = set()
                            st.session_state.clustering_done = False

                            # НЕ сбрасываем csv_file_key!

                            st.success(f"✅ Интерполяция завершена. Добавлено {len(result_df) - old_count} точек")
                            st.rerun()
                        except Exception as e:
                            st.error(f"❌ Ошибка интерполяции: {e}")
                else:
                    st.warning("⚠️ Нет данных для интерполяции")


        # Фильтрация по акватории
        st.markdown("### 🌊 Акватория")

        aquatoria_bounds_dict = {
            "Японское море": {"lat_min": 34, "lat_max": 52, "lon_min": 130, "lon_max": 142},
            "Охотское море": {"lat_min": 43, "lat_max": 62, "lon_min": 135, "lon_max": 165},
            "Берингов пролив": {"lat_min": 60, "lat_max": 66, "lon_min": -170, "lon_max": -165},
            "Татарский пролив": {"lat_min": 45, "lat_max": 52, "lon_min": 140, "lon_max": 142},
            "Залив Петра Великого": {"lat_min": 42.5, "lat_max": 43.5, "lon_min": 131.5, "lon_max": 132.5}
        }

        aquatoria = st.selectbox(
            "Выберите регион",
            ["Японское море", "Охотское море", "Берингов пролив", "Татарский пролив",
             "Залив Петра Великого", "Пользовательская область"]
        )

        if aquatoria == "Пользовательская область":
            col1, col2 = st.columns(2)
            with col1:
                lat_min = st.number_input("Широта min", value=40.0, format="%.2f")
                lat_max = st.number_input("Широта max", value=50.0, format="%.2f")
            with col2:
                lon_min = st.number_input("Долгота min", value=130.0, format="%.2f")
                lon_max = st.number_input("Долгота max", value=145.0, format="%.2f")
        else:
            bounds = aquatoria_bounds_dict[aquatoria]
            lat_min, lat_max = bounds["lat_min"], bounds["lat_max"]
            lon_min, lon_max = bounds["lon_min"], bounds["lon_max"]
            st.info(f"Границы: широта {lat_min}–{lat_max}, долгота {lon_min}–{lon_max}")

        # Фильтрация по типу судна
        st.markdown("### 🚢 Тип судна")
        if st.session_state.df is not None and 'type' in st.session_state.df.columns:
            ship_types = ['Все типы'] + list(st.session_state.df['type'].unique())
            selected_type = st.selectbox("Выберите тип", ship_types)
        else:
            selected_type = "Все типы"
            st.caption("Колонка 'type' не найдена в данных")

        # Фильтрация по скорости
        st.markdown("### ⚡ Скорость (узлы)")
        speed_range = st.slider("Диапазон скорости", 0, 50, (0, 30))

        # Фильтрация по времени
        st.markdown("### 📅 Временной диапазон")

        # Проверяем, есть ли данные и timestamp
        has_valid_timestamp = False

        if st.session_state.df is not None and len(st.session_state.df) > 0:
            if 'timestamp' in st.session_state.df.columns:
                # Преобразуем timestamp в datetime если нужно
                if not pd.api.types.is_datetime64_any_dtype(st.session_state.df['timestamp']):
                    st.session_state.df['timestamp'] = pd.to_datetime(st.session_state.df['timestamp'], errors='coerce')

                # Удаляем NaT значения для определения диапазона
                valid_timestamps = st.session_state.df['timestamp'].dropna()

                if len(valid_timestamps) > 0:
                    has_valid_timestamp = True
                    min_date = valid_timestamps.min().date()
                    max_date = valid_timestamps.max().date()

                    st.session_state.min_date = min_date
                    st.session_state.max_date = max_date

                    # Создаём два столбца для date_from и date_to
                    col_date1, col_date2 = st.columns(2)
                    with col_date1:
                        date_from = st.date_input(
                            "Дата от",
                            value=min_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="date_from"
                        )
                    with col_date2:
                        date_to = st.date_input(
                            "Дата до",
                            value=max_date,
                            min_value=min_date,
                            max_value=max_date,
                            key="date_to"
                        )

                    # Чекбокс для включения/отключения фильтра по времени
                    use_time_filter = st.checkbox("✅ Применить фильтр по времени", value=False, key="use_time_filter")

                    if use_time_filter and date_from > date_to:
                        st.warning("⚠️ Дата 'от' не может быть позже даты 'до'")
                else:
                    st.caption("ℹ️ В данных нет корректных временных меток")
                    use_time_filter = False
                    date_from = None
                    date_to = None
            else:
                st.caption("ℹ️ Колонка 'timestamp' не найдена в данных")
                use_time_filter = False
                date_from = None
                date_to = None
        else:
            st.caption("ℹ️ Нет данных для фильтрации по времени")
            use_time_filter = False
            date_from = None
            date_to = None

        col_filter1, col_filter2 = st.columns(2)
        with col_filter1:
            if st.button("🔍 Применить фильтры", use_container_width=True):
                if st.session_state.df_original is None:
                    st.warning("⚠️ Нет данных для фильтрации. Сначала загрузите данные.")
                else:
                    df_filtered = st.session_state.df_original.copy()

                    # Фильтр по области
                    df_filtered = df_filtered[
                        (df_filtered['lat'] >= lat_min) &
                        (df_filtered['lat'] <= lat_max) &
                        (df_filtered['lon'] >= lon_min) &
                        (df_filtered['lon'] <= lon_max)
                        ]

                    # Фильтр по типу судна
                    if selected_type != "Все типы" and 'type' in df_filtered.columns:
                        df_filtered = df_filtered[df_filtered['type'] == selected_type]

                    # Фильтр по скорости
                    if 'sog' in df_filtered.columns:
                        df_filtered = df_filtered[
                            (df_filtered['sog'] >= speed_range[0]) &
                            (df_filtered['sog'] <= speed_range[1])
                            ]

                    if use_time_filter and date_from and date_to:
                        if 'timestamp' in df_filtered.columns:
                            # Преобразуем в datetime если нужно
                            if not pd.api.types.is_datetime64_any_dtype(df_filtered['timestamp']):
                                df_filtered['timestamp'] = pd.to_datetime(df_filtered['timestamp'])

                            # Применяем фильтр
                            df_filtered = df_filtered[
                                (df_filtered['timestamp'].dt.date >= date_from) &
                                (df_filtered['timestamp'].dt.date <= date_to)
                                ]
                            st.info(f"📅 Временной диапазон: {date_from} – {date_to}")

                    if len(df_filtered) == 0:
                        st.warning("⚠️ После фильтрации не осталось данных. Попробуйте расширить диапазоны.")
                    else:
                        st.session_state.df = df_filtered
                        st.success(f"✅ После фильтрации: {len(df_filtered)} записей")
                        st.session_state.clustering_done = False
                        st.session_state.cluster_labels = None
                        st.session_state.clustering_metrics = None
                        st.info("🔄 Данные изменены.")

        with col_filter2:
            if st.button("🔄 Сбросить фильтры", use_container_width=True):
                if st.session_state.df_original is None:
                    st.warning("⚠️ Нет данных для сброса.")
                else:
                    st.session_state.df = st.session_state.df_original.copy()
                    st.session_state.clustering_done = False
                    st.session_state.clustering_metrics = None
                    st.session_state.cluster_labels = None
                    st.success("✅ Фильтры сброшены")
                    st.rerun()
        st.markdown("---")

        # ==================== КЛАСТЕРИЗАЦИЯ ====================
        st.markdown("## Алгоритм кластеризации")

        def get_clustering_data():
            if st.session_state.df is not None and len(st.session_state.df) > 0:
                return st.session_state.df, "загруженных данных"
            elif 'streaming_points' in st.session_state and len(st.session_state.streaming_points) > 0:
                return pd.DataFrame(st.session_state.streaming_points), "онлайн-имитации"
            else:
                return None, None

        algorithm = st.radio(
            "Выберите алгоритм",
            ["DBSCAN", "OPTICS", "Иерархическая"],
            horizontal=True
        )
        st.session_state.current_algorithm = algorithm

        st.markdown("### ⚙️ Параметры")

        if algorithm == "DBSCAN":
            eps = st.slider("eps (радиус поиска)", 0.01, 0.5, 0.05, 0.01)
            min_samples = st.slider("min_samples (мин. точек)", 3, 20, 5, 1)
            features = st.multiselect(
                "Признаки для кластеризации",
                ['lat', 'lon', 'sog', 'cog'],
                default=['lat', 'lon']
            )

            if st.button("▶ Запустить DBSCAN", type="primary", use_container_width=True):

                if 'online_sim' in st.session_state and st.session_state.online_sim:
                    st.session_state.online_sim.stop_simulation()
                    st.session_state.online_sim = None
                    st.session_state.sim_active = False
                    st.info("Имитация остановлена для выполнения кластеризации")

                if st.session_state.df is not None and len(st.session_state.df) > 0:
                    df_for_clustering = st.session_state.df
                elif 'streaming_points' in st.session_state and len(st.session_state.streaming_points) > 0:
                    df_for_clustering = pd.DataFrame(st.session_state.streaming_points)
                    st.info("📡 Кластеризация выполняется на данных онлайн-имитации")
                else:
                    st.error("❌ Нет данных для кластеризации.")
                    st.stop()

                with st.spinner("Выполняется кластеризация..."):
                    result_df, labels, metrics = clusterer.dbscan_clustering(
                        df_for_clustering,
                        eps=eps,
                        min_samples=min_samples,
                        features=['lat', 'lon']
                    )
                    st.session_state.df = result_df
                    st.session_state.cluster_labels = labels
                    st.session_state.clustering_metrics = metrics
                    st.session_state.clustering_done = True

                    save_analysis_to_history(
                        algorithm='DBSCAN',
                        params={'eps': eps, 'min_samples': min_samples},
                        metrics=metrics
                    )

                    st.success(f"✅ Кластеризация завершена! Выявлено {metrics['n_clusters']} кластеров")

        elif algorithm == "OPTICS":

            if 'online_sim' in st.session_state and st.session_state.online_sim:
                st.session_state.online_sim.stop_simulation()
                st.session_state.online_sim = None
                st.session_state.sim_active = False
                st.info("Имитация остановлена для выполнения кластеризации")

            min_samples = st.slider("min_samples", 3, 20, 5, 1)
            max_eps = st.slider("max_eps (макс. радиус)", 0.1, 1.0, 0.3, 0.05)
            xi = st.slider("xi (порог извлечения)", 0.01, 0.2, 0.05, 0.01)
            features = st.multiselect(
                "Признаки для кластеризации",
                ['lat', 'lon', 'sog', 'cog'],
                default=['lat', 'lon']
            )

            if st.button("▶ Запустить OPTICS", type="primary", use_container_width=True):
                df_cluster, source = get_clustering_data()
                if df_cluster is None:
                    st.error("❌ Нет данных для кластеризации.")
                else:
                    with st.spinner(f"Выполняется кластеризация OPTICS..."):
                        result_df, labels, metrics = clusterer.optics_clustering(
                            df_cluster, min_samples=min_samples,
                            max_eps=max_eps, xi=xi, features=features
                        )
                        st.session_state.df = result_df
                        st.session_state.cluster_labels = labels
                        st.session_state.clustering_metrics = metrics
                        st.session_state.clustering_done = True

                        save_analysis_to_history(
                            algorithm='OPTICS',
                            params={'min_samples': min_samples, 'max_eps': max_eps, 'xi': xi},
                            metrics=metrics
                        )

                        st.success(f"✅ Кластеризация OPTICS завершена! Выявлено {metrics['n_clusters']} кластеров")

        elif algorithm == "Иерархическая":

            if 'online_sim' in st.session_state and st.session_state.online_sim:
                st.session_state.online_sim.stop_simulation()
                st.session_state.online_sim = None
                st.session_state.sim_active = False
                st.info("Имитация остановлена для выполнения кластеризации")

            n_clusters = st.slider("Количество кластеров", 2, 10, 4, 1)
            linkage = st.selectbox("Метод связи", ['ward', 'complete', 'average', 'single'])
            features = st.multiselect(
                "Признаки для кластеризации",
                ['lat', 'lon', 'sog', 'cog'],
                default=['lat', 'lon']
            )

            if st.button("▶ Запустить иерархическую", type="primary", use_container_width=True):
                df_cluster, source = get_clustering_data()
                if df_cluster is None:
                    st.error("❌ Нет данных для кластеризации.")
                else:
                    with st.spinner(f"Выполняется иерархическая кластеризация..."):
                        result_df, labels, metrics = clusterer.hierarchical_clustering(
                            df_cluster, n_clusters=n_clusters, linkage=linkage, features=features
                        )
                        st.session_state.df = result_df
                        st.session_state.cluster_labels = labels
                        st.session_state.clustering_metrics = metrics
                        st.session_state.clustering_done = True

                        save_analysis_to_history(
                            algorithm='Hierarchical',
                            params={'n_clusters': n_clusters, 'linkage': linkage},
                            metrics=metrics
                        )

                        st.success(f"✅ Иерархическая кластеризация завершена! Выявлено {metrics['n_clusters']} кластеров")

        st.markdown("---")

        # ==================== ЭКСПОРТ ====================
        st.markdown("## 💾 Экспорт результатов")

        # Используем 4 разные колонки
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button(" CSV", use_container_width=True):
                if st.session_state.df is not None and len(st.session_state.df) > 0:
                    csv = st.session_state.df.to_csv(index=False, encoding='utf-8-sig').encode('utf-8-sig')
                    st.download_button(
                        "Скачать CSV",
                        csv,
                        "trajectory_data.csv",
                        "text/csv",
                        use_container_width=True
                    )
                    st.success(f"✅ Экспортировано {len(st.session_state.df)} записей")
                else:
                    st.error("❌ Нет данных для экспорта")

        with col2:
            if st.button(" GeoJSON", use_container_width=True):
                if st.session_state.df is not None and len(st.session_state.df) > 0:
                    import tempfile
                    import os

                    temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.geojson', delete=False, encoding='utf-8')
                    temp_path = temp_file.name
                    temp_file.close()

                    export_to_geojson(st.session_state.df, temp_path)

                    with open(temp_path, 'r', encoding='utf-8') as f:
                        geojson_str = f.read()

                    st.download_button(
                        " Скачать GeoJSON",
                        geojson_str,
                        "trajectories.geojson",
                        "application/json",
                        use_container_width=True
                    )
                    os.unlink(temp_path)
                    st.success(f"✅ Экспортировано {len(st.session_state.df)} точек")
                else:
                    st.error("❌ Нет данных для экспорта")

        with col3:
            if st.button(" PNG карты", use_container_width=True):
                from modules.export_utils import export_map_to_png_bytes, is_png_export_available
                from modules.visualization import MapVisualizer
                import folium
                from datetime import datetime

                if not is_png_export_available():
                    st.error("❌ Экспорт в PNG недоступен. Установите: pip install selenium pillow webdriver-manager")
                else:
                    if st.session_state.df is None or len(st.session_state.df) == 0:
                        st.error("❌ Нет данных для отображения на карте")
                    else:
                        with st.spinner("🖼️ Создание изображения..."):
                            # Центр карты
                            center_lat = 42
                            center_lon = 133

                            # Создаём карту
                            export_map = MapVisualizer()
                            export_map.create_base_map(center_lat=center_lat, center_lon=center_lon, zoom=7)

                            # Палитра цветов (как в MapVisualizer)
                            colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                                      '#1abc9c', '#e67e22', '#34495e', '#16a085', '#27ae60']

                            # Проверяем, есть ли кластеризация
                            if 'cluster' in st.session_state.df.columns:
                                # Отображаем с цветами кластеров
                                for cluster_id in st.session_state.df['cluster'].unique():
                                    cluster_data = st.session_state.df[st.session_state.df['cluster'] == cluster_id]


                                    if cluster_id == -1:
                                        color = '#95a5a6'
                                        radius = 2
                                    else:
                                        color = colors[cluster_id % len(colors)]
                                        radius = 3

                                    for _, row in cluster_data.iterrows():
                                        folium.CircleMarker(
                                            location=[row['lat'], row['lon']],
                                            radius=radius,
                                            color=color,
                                            fill=True,
                                            fill_color=color,
                                            fill_opacity=0.7,
                                            weight=1
                                        ).add_to(export_map.map)
                            else:
                                SHIP_COLOR = '#3498db'
                                for _, row in st.session_state.df.iterrows():
                                    folium.CircleMarker(
                                        location=[row['lat'], row['lon']],
                                        radius=2,
                                        color=SHIP_COLOR,
                                        fill=True,
                                        fill_color=SHIP_COLOR,
                                        fill_opacity=0.7,
                                        weight=1
                                    ).add_to(export_map.map)

                            #  PNG байты
                            png_bytes = export_map_to_png_bytes(export_map.map)

                            if png_bytes:
                                st.download_button(
                                    label="📥 Скачать PNG",
                                    data=png_bytes,
                                    file_name=f"map_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                                    mime="image/png",
                                    use_container_width=True
                                )
                                st.success("✅ Карта готова к скачиванию")
                            else:
                                st.error("❌ Не удалось создать изображение")

# ==================== ОСНОВНАЯ ОБЛАСТЬ ====================
if not st.session_state.data_loaded:
    st.info("👈 Загрузите данные или используйте пример в боковой панели")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        ### 📊 Анализ траекторий
        - Кластеризация DBSCAN, OPTICS
        - Иерархическая кластеризация
        - Предобработка
        """)
    with col2:
        st.markdown("""
        ### 🗺️ Визуализация
        - Интерактивные карты
        - Тепловые карты плотности
        - Аналитические графики
        """)
    with col3:
        st.markdown("""
        ### 📈 Экспорт
        - CSV с метками кластеров
        - Геоданные GeoJSON
        - Статистические отчеты
        """)
else:
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        if 'streaming_points' in st.session_state and st.session_state.streaming_points:
            st.metric("📊 Точек", f"{len(st.session_state.streaming_points):,}")
        elif st.session_state.df is not None:
            st.metric("📊 Записей", f"{len(st.session_state.df):,}")
        else:
            st.metric("📊 Записей", "0")

    with col2:
        if 'streaming_points' in st.session_state and st.session_state.streaming_points:
            unique_ships = len(set(p['mmsi'] for p in st.session_state.streaming_points))
            st.metric("🚢 Судов", unique_ships)
        elif st.session_state.df is not None and 'mmsi' in st.session_state.df.columns:
            st.metric("🚢 Судов", st.session_state.df['mmsi'].nunique())
        else:
            st.metric("🚢 Судов", "—")

    with col3:
        if 'streaming_points' in st.session_state and st.session_state.streaming_points:
            avg_speed = sum(p['sog'] for p in st.session_state.streaming_points) / len(st.session_state.streaming_points)
            st.metric("⚡ Средняя скорость", f"{avg_speed:.1f} уз")
        elif st.session_state.df is not None and 'sog' in st.session_state.df.columns:
            st.metric("⚡ Средняя скорость", f"{st.session_state.df['sog'].mean():.1f} уз")
        else:
            st.metric("⚡ Средняя скорость", "—")

    with col4:
        if st.session_state.clustering_done:
            st.metric("🔵 Кластеров", st.session_state.clustering_metrics.get('n_clusters', '—'))
        else:
            st.metric("🔵 Кластеров", "Не выполнена")

    with col5:
        if st.session_state.get('sim_active', False):
            st.metric("🎬 Режим", "Имитация")
        elif st.session_state.get('ais_active', False):
            st.metric("🌍 Режим", "Онлайн")
        else:
            st.metric("📁 Режим", "Офлайн")

    # ========== ПРИЁМ ДАННЫХ ИЗ ОНЛАЙН-ИМИТАЦИИ ==========
    if 'online_sim' in st.session_state and st.session_state.online_sim:
        packets_received = 0

        # Собираем все новые пакеты
        while True:
            packet = st.session_state.online_sim.get_next_packet()
            if packet is None:
                break
            packets_received += 1

            # Инициализация
            if 'streaming_points' not in st.session_state:
                st.session_state.streaming_points = []
            if 'processed_keys' not in st.session_state:
                st.session_state.processed_keys = set()

            # Защита от дублей и валидация
            packet_key = f"{packet['mmsi']}_{packet['lat']:.4f}_{packet['lon']:.4f}"
            if (packet_key not in st.session_state.processed_keys and
                    -90 <= packet['lat'] <= 90 and -180 <= packet['lon'] <= 180 and
                    0 <= packet['sog'] <= 102.2):
                st.session_state.processed_keys.add(packet_key)
                st.session_state.streaming_points.append(packet.copy())

        # Ограничиваем количество точек
        if len(st.session_state.streaming_points) > 500:
            st.session_state.streaming_points = st.session_state.streaming_points[-500:]

        # Обновляем DataFrame (пока имитация активна)
        if packets_received > 0 and st.session_state.streaming_points:
            temp_df = pd.DataFrame(st.session_state.streaming_points)
            is_valid, msg, validated_df = loader.validate_dataframe(temp_df, "потоковых данных имитации")
            if is_valid:
                st.session_state.df = validated_df
            st.rerun()

    # ========== ПРИЁМ ДАННЫХ ИЗ РЕАЛЬНОГО AIS ==========
    if 'ais_stream' in st.session_state and st.session_state.ais_stream:
        packets_received = 0
        while True:
            packet = st.session_state.ais_stream.get_next_packet()
            if packet is None:
                break
            packets_received += 1

            if 'streaming_points' not in st.session_state:
                st.session_state.streaming_points = []

            packet_key = f"{packet['mmsi']}_{packet['lat']:.4f}_{packet['lon']:.4f}"
            if packet_key not in st.session_state.get('processed_keys', set()):
                if 'processed_keys' not in st.session_state:
                    st.session_state.processed_keys = set()

                if -90 <= packet['lat'] <= 90 and -180 <= packet['lon'] <= 180:
                    if 0 <= packet['sog'] <= 102.2:
                        st.session_state.processed_keys.add(packet_key)
                        st.session_state.streaming_points.append(packet.copy())

            if len(st.session_state.streaming_points) > 500:
                st.session_state.streaming_points = st.session_state.streaming_points[-500:]

        if packets_received > 0:
            if st.session_state.streaming_points:
                temp_df = pd.DataFrame(st.session_state.streaming_points)
                if 'timestamp' not in temp_df.columns:
                    temp_df['timestamp'] = pd.Timestamp.now()

                is_valid, msg, validated_df = loader.validate_dataframe(temp_df, "реальных AIS данных")
                if is_valid:
                    st.session_state.df = validated_df

            if st.session_state.get('auto_refresh_ais', False):
                refresh_interval = st.session_state.get('refresh_interval_ais', 3)
                time.sleep(refresh_interval)
                st.rerun()

    st.markdown("---")

    # Вкладки
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["Карта траекторий", "Аналитика", "Данные", "Динамика", " История анализов"])

    with tab1:
        st.markdown("### Интерактивная карта траекторий")

        map_viz = MapVisualizer()

        center_lat = 43.1
        center_lon = 131.9
        zoom_level = 8

        if 'streaming_points' in st.session_state and st.session_state.streaming_points:
            avg_lat = sum(p['lat'] for p in st.session_state.streaming_points) / len(st.session_state.streaming_points)
            avg_lon = sum(p['lon'] for p in st.session_state.streaming_points) / len(st.session_state.streaming_points)
            if 20 < avg_lat < 80 and 100 < avg_lon < 190:
                center_lat = avg_lat
                center_lon = avg_lon
                zoom_level = 7

        map_viz.create_base_map(center_lat=center_lat, center_lon=center_lon, zoom=zoom_level)

        # 2. Отображаем кластеризованные данные (если есть)
        if st.session_state.df is not None and len(st.session_state.df) > 0:
            if 'cluster' in st.session_state.df.columns:
                n_clusters = map_viz.add_cluster_points(
                    st.session_state.df,
                    cluster_col='cluster',
                    point_radius=3,
                    opacity=0.7
                )
                st.success(f"🎨 Отображено {n_clusters} кластеров")
            else:
                points_shown = map_viz.add_ship_trajectories(
                    st.session_state.df,
                    max_points=3000,
                    ship_color='#3498db'
                )
                st.info(f"ℹ️ Отображено {points_shown} точек. Запустите кластеризацию для цветного отображения")

        folium_static(map_viz.map, width=1900, height=1000)

    with tab2:
        st.markdown("### Аналитика и метрики")

        has_cluster = 'cluster' in st.session_state.df.columns if st.session_state.df is not None else False

        if st.session_state.clustering_done and st.session_state.clustering_metrics:
            metrics = st.session_state.clustering_metrics

            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("🔵 Кластеров", metrics.get('n_clusters', '—'))
            with col2:
                st.metric("⚪ Шум (аномалии)", metrics.get('n_noise', '—'))
            with col3:
                sil_score = metrics.get('silhouette_score')
                st.metric("📈 Silhouette Score", f"{sil_score:.3f}" if sil_score else "N/A")
            with col4:
                db_score = metrics.get('davies_bouldin_score')
                st.metric("📊 Davies-Bouldin", f"{db_score:.3f}" if db_score else "N/A")

            st.markdown("---")

            if has_cluster:

                chart_viz = ChartVisualizer(style='plotly')

                try:
                    fig1, fig2 = chart_viz.plot_cluster_distribution(st.session_state.df)
                    col1, col2 = st.columns(2)
                    with col1:
                        st.plotly_chart(fig1, use_container_width=True)
                    with col2:
                        st.plotly_chart(fig2, use_container_width=True)
                except Exception as e:
                    st.warning(f"Ошибка графика распределения: {e}")

                if 'sog' in st.session_state.df.columns:
                    try:
                        fig2 = chart_viz.plot_speed_by_cluster(st.session_state.df)
                        st.plotly_chart(fig2, use_container_width=True)
                    except Exception as e:
                        st.warning(f"Ошибка графика скоростей: {e}")

                if st.session_state.current_algorithm == "Иерархическая":
                    st.markdown("---")
                    st.markdown("### Дендрограмма иерархической кластеризации")

                    try:
                        # Получаем данные для дендрограммы
                        df_for_dendrogram = st.session_state.df.copy()

                        # Исключаем шум для дендрограммы (опционально)
                        df_for_dendrogram = df_for_dendrogram[df_for_dendrogram['cluster'] != -1]

                        if len(df_for_dendrogram) > 1:
                            with st.spinner("Построение дендрограммы..."):
                                # Получаем метод связи из метрик
                                linkage_method = st.session_state.clustering_metrics.get('linkage', 'ward')

                                fig_dendro = clusterer.plot_dendrogram(
                                    df_for_dendrogram,
                                    features=['lat', 'lon'],
                                    linkage_method=linkage_method,
                                    figsize=(12, 6)
                                )
                                st.pyplot(fig_dendro)
                                plt.close(fig_dendro)
                        else:
                            st.info("Недостаточно данных для построения дендрограммы (нужно > 1 точки)")
                    except Exception as e:
                        st.warning(f"Ошибка построения дендрограммы: {e}")

            else:
                st.error("❌ Колонка 'cluster' не найдена в данных!")
        else:
            st.info(" Запустите кластеризацию в боковой панели")

    with tab3:
        st.markdown("### Данные траекторий")

        # Определяем, какие данные показывать
        display_df = None

        # Приоритет: сначала загруженные данные, потом точки имитации
        if st.session_state.df is not None and len(st.session_state.df) > 0:
            display_df = st.session_state.df
            data_source_text = "📊 Загруженные данные"
        elif 'streaming_points' in st.session_state and len(st.session_state.streaming_points) > 0:
            display_df = pd.DataFrame(st.session_state.streaming_points)
            data_source_text = "🚢 Данные онлайн-имитации / реального AIS"
        else:
            display_df = None
            data_source_text = "📊 Данные"

        if display_df is not None and len(display_df) > 0:
            st.markdown(f"#### {data_source_text}")

            # Слайдер для количества строк
            rows_limit = st.slider(
                "Количество строк для отображения",
                min_value=10, max_value=500, value=100, step=10,
                key="rows_limit_tab3"
            )

            # Выбор колонок для отображения
            default_cols = ['mmsi', 'lat', 'lon', 'sog', 'type']
            if 'cluster' in display_df.columns:
                default_cols.append('cluster')

            columns_to_show = st.multiselect(
                "Выберите колонки для отображения",
                display_df.columns.tolist(),
                default=[col for col in default_cols if col in display_df.columns]
            )

            if columns_to_show:
                st.dataframe(display_df[columns_to_show].head(rows_limit), use_container_width=True)
                st.caption(f"Показано {min(len(display_df), rows_limit)} из {len(display_df)} записей")
            else:
                st.info("ℹ️ Выберите хотя бы одну колонку для отображения")
        else:
            st.info("ℹ️ Нет данных для отображения. Загрузите данные или запустите имитацию.")

    with tab4:
        st.markdown("### 📈 Динамика движения судов")

        if st.session_state.df is not None and len(st.session_state.df) > 0:
            chart_viz = ChartVisualizer(style='plotly')

            # Выбор типа группировки
            group_by = st.radio(
                "Группировка по:",
                ["hour", "day", "month"],
                format_func=lambda x: {"hour": "Часам", "day": "Дням", "month": "Месяцам"}[x],
                horizontal=True
            )

            fig = chart_viz.plot_temporal_density(st.session_state.df, group_by=group_by)

            if fig:
                st.plotly_chart(fig, use_container_width=True)

                # Дополнительная статистика
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Всего сообщений", f"{len(st.session_state.df):,}")
                with col2:
                    if 'timestamp' in st.session_state.df.columns:
                        time_min = pd.to_datetime(st.session_state.df['timestamp']).min()
                        st.metric("Начало периода", time_min.strftime('%Y-%m-%d'))
                with col3:
                    if 'timestamp' in st.session_state.df.columns:
                        time_max = pd.to_datetime(st.session_state.df['timestamp']).max()
                        st.metric("Конец периода", time_max.strftime('%Y-%m-%d'))
        else:
            st.info(" Сначала загрузите данные для отображения динамики")

    with tab5:
        st.markdown("### История выполненных анализов")

        if len(st.session_state.analysis_history) == 0:
            st.info(" История пуста. Выполните кластеризацию для сохранения результатов.")
        else:
            # Кнопки управления
            col1, col2 = st.columns(2)
            with col1:
                if st.button("🗑️ Очистить историю", use_container_width=True):
                    st.session_state.analysis_history = []
                    st.rerun()
            with col2:
                if st.button("📊 Сравнить метрики", use_container_width=True):
                    st.session_state.show_history_charts = True

            st.markdown("---")

            # Преобразуем историю в DataFrame для отображения
            history_df = pd.DataFrame(st.session_state.analysis_history)


            # Форматируем параметры для красивого отображения
            def format_params(params):
                return ', '.join([f"{k}={v}" for k, v in params.items()])


            history_df['params_str'] = history_df['params'].apply(format_params)

            # Выбираем колонки для отображения
            display_columns = ['timestamp', 'algorithm', 'params_str', 'n_clusters', 'n_noise', 'silhouette_score',
                               'davies_bouldin_score']
            display_names = {
                'timestamp': 'Время',
                'algorithm': 'Алгоритм',
                'params_str': 'Параметры',
                'n_clusters': 'Кластеров',
                'n_noise': 'Шум',
                'silhouette_score': 'Silhouette',
                'davies_bouldin_score': 'Davies-Bouldin'
            }

            # Форматируем метрики
            history_df['silhouette_score'] = history_df['silhouette_score'].apply(lambda x: f"{x:.3f}" if x else "N/A")
            history_df['davies_bouldin_score'] = history_df['davies_bouldin_score'].apply(
                lambda x: f"{x:.3f}" if x else "N/A")

            st.dataframe(
                history_df[display_columns].rename(columns=display_names),
                use_container_width=True,
                height=400
            )

            # График сравнения метрик (если нажата кнопка)
            if st.session_state.get('show_history_charts', False):
                import plotly.graph_objects as go

                st.markdown("---")
                st.markdown("### 📈 Сравнение метрик по запускам")

                # Убираем записи без silhouette_score
                chart_df = history_df[history_df['silhouette_score'] != 'N/A'].copy()

                if len(chart_df) > 1:
                    chart_df['silhouette_score'] = pd.to_numeric(chart_df['silhouette_score'])
                    chart_df['index'] = range(len(chart_df))

                    fig = go.Figure()
                    fig.add_trace(go.Scatter(
                        x=chart_df['index'],
                        y=chart_df['silhouette_score'],
                        mode='lines+markers',
                        name='Silhouette Score',
                        line=dict(color='#3498db', width=2),
                        marker=dict(size=8)
                    ))

                    fig.update_layout(
                        title='Динамика качества кластеризации',
                        xaxis_title='Номер запуска (от свежего к старому)',
                        yaxis_title='Silhouette Score',
                        height=400,
                        template='plotly_white'
                    )

                    st.plotly_chart(fig, use_container_width=True)

                    # Подписи к точкам
                    st.caption(" Каждая точка на графике соответствует одному запуску кластеризации")
                else:
                    st.info(" Недостаточно данных для построения графика (нужно минимум 2 записи)")

                # Кнопка скрыть график
                if st.button("Скрыть график"):
                    st.session_state.show_history_charts = False
                    st.rerun()

# ========== АВТООБНОВЛЕНИЕ ПО ТАЙМЕРУ ==========
if st.session_state.get('sim_active', False) and st.session_state.get('auto_refresh', False):
    refresh_interval = st.session_state.get('refresh_interval', 3)
    st.info(f"🟢 Автообновление ВКЛЮЧЕНО (каждые {refresh_interval} сек)")
    time.sleep(refresh_interval)
    st.rerun()

# Footer
st.markdown("---")
st.markdown(
    "<p style='text-align: center; color: #6c7a8a; font-size: 12px;'>"
    "© 2026 Кластерный анализ траекторных данных | Дальневосточный федеральный университет | ИМиКТ"
    "</p>",
    unsafe_allow_html=True
)