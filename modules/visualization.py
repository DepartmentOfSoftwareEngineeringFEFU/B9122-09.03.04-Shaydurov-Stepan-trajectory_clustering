"""
Модуль визуализации траекторных данных
Создание интерактивных карт и графиков
"""

import folium
import pandas as pd
import numpy as np
from folium import plugins
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from branca.colormap import linear
import streamlit as st




def save_analysis_to_history(algorithm: str, params: dict, metrics: dict):
    """Сохраняет результат кластеризации в историю"""
    from datetime import datetime

    record = {
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
        'algorithm': algorithm,
        'params': params,
        'n_clusters': metrics.get('n_clusters', 0),
        'n_noise': metrics.get('n_noise', 0),
        'silhouette_score': metrics.get('silhouette_score', None),
        'davies_bouldin_score': metrics.get('davies_bouldin_score', None),
        'n_samples': metrics.get('n_samples', 0)
    }

    # Добавляем в начало списка (свежие сверху)
    st.session_state.analysis_history.insert(0, record)

    # Ограничиваем историю 20 записями
    if len(st.session_state.analysis_history) > 20:
        st.session_state.analysis_history = st.session_state.analysis_history[:20]

class MapVisualizer:
    """Класс для создания интерактивных карт с траекториями"""

    def __init__(self):
        self.map = None
        self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                       '#1abc9c', '#e67e22', '#34495e', '#16a085', '#27ae60']

    def create_base_map(self, center_lat=45.0, center_lon=135.0, zoom=6):
        """
        Создание базовой карты

        Parameters:
        -----------
        center_lat, center_lon : float
            Центр карты
        zoom : int
            Уровень масштабирования

        Returns:
        --------
        map : folium.Map
            Базовая карта
        """
        self.map = folium.Map(
            location=[center_lat, center_lon],
            zoom_start=zoom,
            tiles='CartoDB positron',
            control_scale=True
        )

        # Добавление полноэкранного режима
        plugins.Fullscreen().add_to(self.map)

        # Добавление слоя с улицами
        folium.TileLayer(
            tiles='OpenStreetMap',
            name='OpenStreetMap',
            show=False
        ).add_to(self.map)

        # Добавление спутникового слоя
        folium.TileLayer(
            tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr='Esri',
            name='Спутник',
            show=False
        ).add_to(self.map)

        # Добавление контроля слоев
        folium.LayerControl().add_to(self.map)

        return self.map


    def add_ship_trajectories(self, df, max_points=3000, ship_color='#3498db',
                              lat_col='lat', lon_col='lon',
                              time_cols=['timestamp', 'date_add'],
                              id_cols=['mmsi', 'id_marine', 'id_track'],
                              point_radius=3, line_weight=2, opacity=0.7):
        """
        Добавление траекторий и точек судов на карту

        Parameters:
        -----------
        df : DataFrame
            Данные с координатами
        max_points : int
            Максимальное количество отображаемых точек (для производительности)
        ship_color : str
            Цвет судов (в hex-формате)
        lat_col, lon_col : str
            Названия колонок с координатами
        time_cols : list
            Возможные названия колонок с временем (в порядке приоритета)
        id_cols : list
            Возможные названия колонок с идентификатором судна
        point_radius : int
            Радиус точек-маркеров
        line_weight : int
            Толщина линии траектории
        opacity : float
            Прозрачность (0-1)

        Returns:
        --------
        int : Количество отображённых точек
        """
        if self.map is None:
            self.create_base_map()

        if df is None or df.empty:
            print("⚠️ Нет данных для отображения")
            return 0

        # Ограничиваем количество точек
        original_len = len(df)
        if len(df) > max_points:
            df_display = df.head(max_points)
            print(f"⚠️ Отображается {max_points} из {original_len} точек")
        else:
            df_display = df.copy()

        # Определяем колонку с идентификатором судна
        ship_id_col = None
        for col in id_cols:
            if col in df_display.columns:
                ship_id_col = col
                break

        if ship_id_col is None:
            print("⚠️ Не найдена колонка с идентификатором судна")
            return 0

        # Определяем колонку с временем
        time_col = None
        for col in time_cols:
            if col in df_display.columns:
                time_col = col
                break

        # Группируем точки по судам
        ships_points = {}
        for _, row in df_display.iterrows():
            ship_id = row.get(ship_id_col, 0)
            if ship_id not in ships_points:
                ships_points[ship_id] = []
            ships_points[ship_id].append(row)

        # Отрисовка для каждого судна
        for ship_id, points in ships_points.items():
            # Сортировка по времени
            if time_col:
                sorted_points = sorted(points, key=lambda x: x.get(time_col, ''))
            else:
                sorted_points = points

            # Рисуем траекторию (линию)
            if len(sorted_points) > 1:
                trajectory_coords = [[p[lat_col], p[lon_col]] for p in sorted_points]
                folium.PolyLine(
                    trajectory_coords,
                    color=ship_color,
                    weight=line_weight,
                    opacity=opacity,
                    popup=f"🚢 Судно {ship_id}"
                ).add_to(self.map)

            # Рисуем точки
            for point in points:
                lat = point.get(lat_col)
                lon = point.get(lon_col)
                if lat is not None and lon is not None:
                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=point_radius,
                        color=ship_color,
                        fill=True,
                        fill_color=ship_color,
                        fill_opacity=0.7,
                        weight=1,
                        popup=self._create_ship_popup(ship_id, point)
                    ).add_to(self.map)

        return len(df_display)

    def _create_ship_popup(self, ship_id, point):
        """
        Создание HTML-содержимого для всплывающей подсказки судна

        Parameters:
        -----------
        ship_id : int/str
            Идентификатор судна
        point : Series/dict
            Точка с данными

        Returns:
        --------
        str : HTML-строка для popup
        """
        lat = point.get('lat', 0)
        lon = point.get('lon', 0)
        sog = point.get('sog', 0)

        return f"""🚢 Судно {ship_id}<br>
                   📍 {lat:.4f}, {lon:.4f}<br>
                   ⚡ {sog} уз"""

    def add_cluster_points(self, df, cluster_col='cluster',
                           lat_col='lat', lon_col='lon',
                           point_radius=3, opacity=0.7):
        """
        Добавление всех точек кластеров с цветовой кодировкой И траекторий
        """
        if self.map is None:
            self.create_base_map()

        if df is None or df.empty:
            print("⚠️ Нет данных для отображения")
            return 0

        if cluster_col not in df.columns:
            print(f"⚠️ Колонка '{cluster_col}' не найдена")
            return 0

        all_clusters = df[cluster_col].unique()
        cluster_ids = [c for c in all_clusters if c != -1]
        n_clusters = len(cluster_ids)
        print(f"🎨 Отображение {n_clusters} кластеров")

        # ========== РИСУЕМ ТРАЕКТОРИИ ДЛЯ КАЖДОГО КЛАСТЕРА ==========
        for cluster_id in cluster_ids:
            cluster_data = df[df[cluster_col] == cluster_id]
            color = self.colors[cluster_id % len(self.colors)]
            label = f"Кластер {cluster_id + 1}"

            # Группируем по судам внутри кластера
            if 'mmsi' in cluster_data.columns:
                for mmsi, ship_data in cluster_data.groupby('mmsi'):
                    if 'timestamp' in ship_data.columns:
                        ship_data = ship_data.sort_values('timestamp')
                    points = list(zip(ship_data[lat_col], ship_data[lon_col]))

                    # Рисуем линию траектории
                    if len(points) > 1:
                        folium.PolyLine(
                            points,
                            color=color,
                            weight=2,
                            opacity=0.7,
                            popup=f"{label} | Судно {mmsi}"
                        ).add_to(self.map)

            # Рисуем точки кластера
            for _, row in cluster_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=point_radius,
                    color=color,
                    fill=True,
                    fill_color=color,
                    fill_opacity=opacity,
                    popup=f"{label}<br>Скорость: {row.get('sog', '?')} уз"
                ).add_to(self.map)

        # ========== ШУМ (точки без линий) ==========
        if -1 in all_clusters:
            noise_data = df[df[cluster_col] == -1]
            for _, row in noise_data.iterrows():
                folium.CircleMarker(
                    location=[row[lat_col], row[lon_col]],
                    radius=2,
                    color='#95a5a6',
                    fill=True,
                    fill_color='#95a5a6',
                    fill_opacity=0.5,
                    popup="Шум (аномалия)"
                ).add_to(self.map)

        return n_clusters

    def display_vessels_and_clusters(self, df, max_points=3000,
                                     show_vessels=True, show_clusters=True,
                                     ship_color='#3498db', cluster_point_radius=1):
        """
        Комбинированный метод для отображения судов и кластеров

        Parameters:
        -----------
        df : DataFrame
            Данные с координатами
        max_points : int
            Максимальное количество отображаемых точек
        show_vessels : bool
            Показывать ли траектории судов
        show_clusters : bool
            Показывать ли кластеризацию
        ship_color : str
            Цвет судов
        cluster_point_radius : int
            Радиус точек кластеров

        Returns:
        --------
        dict : Статистика отображения
        """
        result = {'vessels_shown': 0, 'clusters_shown': 0}

        if show_vessels:
            result['vessels_shown'] = self.add_ship_trajectories(
                df, max_points=max_points, ship_color=ship_color
            )

        if show_clusters and 'cluster' in df.columns:
            result['clusters_shown'] = self.add_cluster_points(
                df, point_radius=cluster_point_radius
            )

        return result


class ChartVisualizer:
    """Класс для создания графиков и диаграмм"""

    def __init__(self, style='plotly'):
        self.style = style
        # ========== ИСПРАВЛЕНИЕ: используем ту же палитру, что и на карте ==========
        self.colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                       '#1abc9c', '#e67e22', '#34495e', '#16a085', '#27ae60']

    def plot_temporal_density(self, df, time_col='timestamp', group_by='hour'):
        """
        График динамики плотности движения во времени

        Parameters:
        -----------
        df : pd.DataFrame
            Данные с колонкой timestamp
        time_col : str
            Название колонки с временем
        group_by : str
            Группировка: 'hour', 'day', 'month'

        Returns:
        --------
        plotly.graph_objects.Figure или None
        """
        if time_col not in df.columns:
            return None

        df_plot = df.copy()

        # Преобразуем в datetime если нужно
        if not pd.api.types.is_datetime64_any_dtype(df_plot[time_col]):
            df_plot[time_col] = pd.to_datetime(df_plot[time_col], errors='coerce')

        # Удаляем строки с некорректным временем
        df_plot = df_plot.dropna(subset=[time_col])

        if df_plot.empty:
            return None

        if group_by == 'hour':
            df_plot['group'] = df_plot[time_col].dt.hour
            title = 'Динамика плотности движения по часам суток'
            xlabel = 'Час'
        elif group_by == 'day':
            df_plot['group'] = df_plot[time_col].dt.date
            title = 'Динамика плотности движения по дням'
            xlabel = 'Дата'
        else:
            df_plot['group'] = df_plot[time_col].dt.month
            title = 'Динамика плотности движения по месяцам'
            xlabel = 'Месяц'

        # Подсчёт количества точек в каждой группе
        density = df_plot.groupby('group').size().reset_index(name='count')

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=density['group'],
            y=density['count'],
            marker_color='#3498db',
            name='Количество сообщений'
        ))

        fig.update_layout(
            title=title,
            xaxis_title=xlabel,
            yaxis_title='Количество AIS сообщений',
            height=400,
            template='plotly_white'
        )

        return fig

    def plot_cluster_distribution(self, df, cluster_col='cluster'):
        """
        Диаграмма распределения точек по кластерам
        """
        cluster_counts = df[cluster_col].value_counts().sort_index()

        # Разделение на кластеры и шум
        cluster_data = {k: v for k, v in cluster_counts.items() if k != -1}
        noise_count = cluster_counts.get(-1, 0)

        # ========== ИСПРАВЛЕНИЕ: сортируем кластеры для соответствия цветов ==========
        sorted_clusters = sorted(cluster_data.keys())

        if self.style == 'plotly':
            fig1 = go.Figure()
            fig1.add_trace(go.Bar(
                x=list(sorted_clusters),
                y=[cluster_data[c] for c in sorted_clusters],
                name='Кластеры',
                marker_color=[self.colors[c % len(self.colors)] for c in sorted_clusters]
            ))
            fig1.update_layout(
                title='Распределение по кластерам',
                xaxis_title='Кластер',
                yaxis_title='Количество точек',
                height=400
            )

            fig2 = go.Figure()
            fig2.add_trace(go.Pie(
                labels=['Кластеры', 'Шум'],
                values=[sum(cluster_data.values()), noise_count],
                marker_colors=['#2ecc71', '#95a5a6'],
                hole=0.3
            ))
            fig2.update_layout(
                title='Кластеры vs Шум',
                height=400
            )

            return fig1, fig2
        else:
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

            bars = ax1.bar(
                list(sorted_clusters),
                [cluster_data[c] for c in sorted_clusters],
                color=[self.colors[c % len(self.colors)] for c in sorted_clusters]
            )
            ax1.set_xlabel('Кластер')
            ax1.set_ylabel('Количество точек')
            ax1.set_title('Распределение по кластерам')

            ax2.pie([sum(cluster_data.values()), noise_count],
                    labels=['Кластеры', 'Шум'],
                    autopct='%1.1f%%',
                    colors=['#2ecc71', '#95a5a6'])
            ax2.set_title('Кластеры vs Шум')

            return fig

    def plot_speed_by_cluster(self, df, speed_col='sog', cluster_col='cluster'):
        """
        Boxplot скоростей по кластерам
        """
        # Исключаем шум для чистоты
        plot_data = df[df[cluster_col] != -1].copy()

        # Сортируем кластеры
        plot_data = plot_data.sort_values(cluster_col)

        if self.style == 'plotly':
            fig = px.box(plot_data, x=cluster_col, y=speed_col,
                         title='Распределение скоростей по кластерам',
                         labels={cluster_col: 'Кластер', speed_col: 'Скорость (узлы)'},
                         color=cluster_col,
                         color_discrete_sequence=self.colors)
            return fig
        else:
            fig, ax = plt.subplots(figsize=(10, 6))
            clusters = sorted(plot_data[cluster_col].unique())
            data_to_plot = [plot_data[plot_data[cluster_col] == c][speed_col] for c in clusters]

            bp = ax.boxplot(data_to_plot, labels=clusters, patch_artist=True)
            for patch, idx in zip(bp['boxes'], range(len(clusters))):
                patch.set_facecolor(self.colors[clusters[idx] % len(self.colors)])

            ax.set_xlabel('Кластер')
            ax.set_ylabel('Скорость (узлы)')
            ax.set_title('Распределение скоростей по кластерам')
            return fig