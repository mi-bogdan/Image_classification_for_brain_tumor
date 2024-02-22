FROM python:3.11

WORKDIR /app


COPY requirements.txt /app/
RUN pip install -r requirements.txt

# Устанавливаем зависимости для OpenCV
RUN apt-get update && apt-get install -y \
    libgl1-mesa-dev \
    libglib2.0-0

COPY . /app

# Устанавливаем переменную окружения для streamlit
ENV STREAMLIT_SERVER_PORT=8501

EXPOSE 8501
# Команда для запуска Streamlit
CMD ["streamlit", "run", "main.py", "--server.address=0.0.0.0"]