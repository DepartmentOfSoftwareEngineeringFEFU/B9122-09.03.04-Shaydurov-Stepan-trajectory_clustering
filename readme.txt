Способ 1: Запуск через терминал
Для стабильное работы требуется версия Python 3.11
 
# 1. Создание виртуального окружения
python -m venv venv

# 2. Активация виртуального окружения
# Windows
venv\Scripts\activate

   #Linux/macOS
   source venv/bin/activate

# 3. Установка зависимостей
pip install -r requirements.txt

# 4. Запуск приложения
streamlit run app.py

# 5. Запуск приложения
http://localhost:8501

Способ 2: Запуск через Docker

# Сборка образа и запуск контейнера
docker-compose up -d

# Просмотр логов (для проверки работы)
docker-compose logs -f

# Остановка контейнера
docker-compose down