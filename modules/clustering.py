import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN, OPTICS, AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
from scipy.cluster.hierarchy import linkage, dendrogram
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


class ClusterAnalyzer:
    """Класс для выполнения кластерного анализа траекторных данных"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.labels_ = None
        self.model_ = None
        self.metrics_ = {}

    def preprocess_features(self, df, features=['lat', 'lon', 'sog', 'cog']):
        """
        Подготовка признаков для кластеризации
        Parameters:
        -----------
        df : DataFrame
            Данные с колонками lat, lon, sog (скорость), cog (курс)
        features : list
            Список признаков для кластеризации
        Returns:
        --------
        X : numpy array
            Нормализованные признаки
        """
        # Проверка наличия колонок
        available_features = [f for f in features if f in df.columns]

        if not available_features:
            raise ValueError("Нет доступных признаков для кластеризации")

        # Извлечение признаков
        X = df[available_features].values

        # Обработка пропусков
        X = np.nan_to_num(X)

        # Нормализация
        X_scaled = self.scaler.fit_transform(X)

        return X_scaled

    def dbscan_clustering(self, df, eps=0.05, min_samples=5, features=['lat', 'lon', 'sog', 'cog']):

        # подготовка данных
        X = self.preprocess_features(df, features)

        #  кластеризация
        self.model_ = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = self.model_.fit_predict(X)

        # расчет метрик
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        self.metrics_ = {
            'algorithm': 'DBSCAN',
            'eps': eps,
            'min_samples': min_samples,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_samples': len(self.labels_)
        }

        # Расчет silhouette_score
        if n_clusters >= 2 and n_noise < len(self.labels_) * 0.5:
            try:
                self.metrics_['davies_bouldin_score'] = davies_bouldin_score(X, self.labels_)
                mask = self.labels_ != -1
                if mask.sum() > 0:
                    self.metrics_['silhouette_score'] = silhouette_score(
                        X[mask], self.labels_[mask]
                    )
            except:
                self.metrics_['silhouette_score'] = None
                self.metrics_['davies_bouldin_score'] = None
        else:
            self.metrics_['silhouette_score'] = None
            self.metrics_['davies_bouldin_score'] = None

        # ВАЖНО: создаем копию и добавляем колонку cluster
        df_result = df.copy()
        df_result.loc[:, 'cluster'] = self.labels_

        # Проверка: убедимся что колонка добавилась
        print(f"Колонки в результате: {df_result.columns.tolist()}")
        print(f"Уникальные кластеры: {df_result['cluster'].unique()}")

        return df_result, self.labels_, self.metrics_

    def optics_clustering(self, df, min_samples=5, max_eps=0.3, xi=0.05, features=['lat', 'lon', 'sog', 'cog']):
        """
        Кластеризация методом OPTICS
        """
        # Подготовка данных
        X = self.preprocess_features(df, features)

        # Выполнение кластеризации
        self.model_ = OPTICS(min_samples=min_samples, max_eps=max_eps, xi=xi)
        self.labels_ = self.model_.fit_predict(X)

        # Расчет метрик
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)

        self.metrics_ = {
            'algorithm': 'OPTICS',
            'min_samples': min_samples,
            'max_eps': max_eps,
            'xi': xi,
            'n_clusters': n_clusters,
            'n_noise': n_noise,
            'n_samples': len(self.labels_)
        }

        # Расчет silhouette_score
        if n_clusters >= 2 and n_noise < len(self.labels_) * 0.5:
            try:
                self.metrics_['davies_bouldin_score'] = davies_bouldin_score(X, self.labels_)
                mask = self.labels_ != -1
                if mask.sum() > 0:
                    self.metrics_['silhouette_score'] = silhouette_score(
                        X[mask], self.labels_[mask]
                    )
            except:
                self.metrics_['davies_bouldin_score'] = None
                self.metrics_['silhouette_score'] = None
        else:
            self.metrics_['silhouette_score'] = None
            self.metrics_['davies_bouldin_score'] = None

        # Добавление меток в DataFrame
        df_result = df.copy()
        df_result['cluster'] = self.labels_

        return df_result, self.labels_, self.metrics_

    def hierarchical_clustering(self, df, n_clusters=4, linkage='ward',
                                features=['lat', 'lon', 'sog', 'cog']):
        """
        Иерархическая кластеризация
        """
        # Подготовка данных
        X = self.preprocess_features(df, features)

        # Выполнение кластеризации
        self.model_ = AgglomerativeClustering(n_clusters=n_clusters, linkage=linkage)
        self.labels_ = self.model_.fit_predict(X)

        # Расчет метрик
        n_clusters_found = len(set(self.labels_))

        self.metrics_ = {
            'algorithm': 'Hierarchical',
            'n_clusters': n_clusters_found,
            'linkage': linkage,
            'n_samples': len(self.labels_)
        }

        # Расчет silhouette_score
        if n_clusters_found >= 2:
            try:
                self.metrics_['davies_bouldin_score'] = davies_bouldin_score(X, self.labels_)
                self.metrics_['silhouette_score'] = silhouette_score(X, self.labels_)
            except:
                self.metrics_['silhouette_score'] = None
                self.metrics_['davies_bouldin_score'] = None

        # Добавление меток в DataFrame
        df_result = df.copy()
        df_result['cluster'] = self.labels_

        return df_result, self.labels_, self.metrics_
    def get_cluster_statistics(self, df):
        """
        Получение статистики по кластерам

        Parameters:
        -----------
        df : DataFrame
            Данные с колонкой 'cluster'

        Returns:
        --------
        stats : DataFrame
            Статистика по кластерам
        """
        if 'cluster' not in df.columns:
            raise ValueError("DataFrame не содержит колонку 'cluster'")

        stats = []
        for cluster_id in sorted(df['cluster'].unique()):
            cluster_data = df[df['cluster'] == cluster_id]
            is_noise = (cluster_id == -1)

            stat = {
                'cluster_id': cluster_id,
                'type': 'Шум' if is_noise else 'Кластер',
                'count': len(cluster_data),
                'percentage': len(cluster_data) / len(df) * 100
            }

            # Добавление средних значений числовых колонок
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            for col in numeric_cols:
                if col != 'cluster':
                    stat[f'avg_{col}'] = cluster_data[col].mean()

            stats.append(stat)

        return pd.DataFrame(stats)

    def plot_dendrogram(self, df, features=['lat', 'lon', 'sog', 'cog'],
                        linkage_method='ward', figsize=(10, 6)):
        """
        Построение дендрограммы для иерархической кластеризации

        Parameters:
        -----------
        df : DataFrame
            Данные для кластеризации
        features : list
            Признаки для кластеризации
        linkage_method : str
            Метод связи ('ward', 'complete', 'average', 'single')
        figsize : tuple
            Размер графика

        Returns:
        --------
        fig : matplotlib.figure.Figure
            Фигура с дендрограммой
        """
        # Подготовка данных
        X = self.preprocess_features(df, features)

        # Вычисление матрицы связей
        linkage_matrix = linkage(X, method=linkage_method)

        # Создание фигуры
        fig, ax = plt.subplots(figsize=figsize)

        # Построение дендрограммы
        dendrogram(
            linkage_matrix,
            ax=ax,
            truncate_mode='level',  # Показывать только верхние уровни
            p=5,  # Количество уровней
            show_leaf_counts=True,  # Показывать количество точек в листьях
            leaf_rotation=90,
            leaf_font_size=10,
            color_threshold=0.7
        )

        ax.set_title('Дендрограмма иерархической кластеризации')
        ax.set_xlabel('Индекс точки (или количество точек в кластере)')
        ax.set_ylabel('Расстояние')

        return fig
