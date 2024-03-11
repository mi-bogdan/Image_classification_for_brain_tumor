import os
import time

import cv2
import joblib
import numpy as np
from prettytable import PrettyTable
from skimage.io import imread
from skimage.transform import resize
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from tqdm import tqdm


def load_images_and_labels(base_dir="Training"):
    """Загрузка изображений"""

    categories = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    images = []
    labels = []
    count = 0

    for label, category in enumerate(categories):
        path = os.path.join(base_dir, category)
        for img_name in tqdm(
            os.listdir(path), desc=f"Загрузка изображений из {category}"
        ):
            img_path = os.path.join(path, img_name)
            img = imread(img_path, as_gray=True)
            img = resize(img, (128, 128), anti_aliasing=True)

            images.append(img)
            labels.append(label)
            count += 1

    return np.array(images), np.array(labels), count


def aggregate_descriptors(descriptors_list):
    """Аггрегация дескрипторов каждого изображения с использованием среднего значения"""
    agg_descriptors = []
    for des in descriptors_list:
        if des is not None:
            agg_descriptors.append(np.mean(des, axis=0))
        else:
            agg_descriptors.append(np.zeros(128))

    return np.array(agg_descriptors)


def extract_sift_descriptors(images):
    """Извлечение дескриптора SIFT"""

    sift = cv2.SIFT_create()
    descriptors_list = []

    for img in tqdm(images, desc="Извлечение дескрипторов SIFT"):
        kp, des = sift.detectAndCompute((img * 255).astype("uint8"), None)
        if des is not None:
            descriptors_list.append(des)

    # Аггрегация дескрипторов каждого изображения

    agg_descriptors = aggregate_descriptors(descriptors_list)

    return agg_descriptors


def evaluate_model(model, X_test, y_test):
    """Статистика по модели"""

    start_time = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)

    recall = recall_score(y_test, predictions, average="weighted")
    precision = precision_score(y_test, predictions, average="weighted")

    return accuracy, recall, precision, predict_time


images, labels, total_images_count = load_images_and_labels()
agg_descriptors = extract_sift_descriptors(images)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    agg_descriptors, labels, test_size=0.3, random_state=42
)

# Обучение классификатора SVM
svm_clf = make_pipeline(StandardScaler(), SVC(kernel="rbf"))
svm_clf.fit(X_train, y_train)

# Сохранение модели
joblib.dump(svm_clf, "svm_clf.joblib")
model_size = os.path.getsize("svm_clf.joblib") / (1024 * 1024)  # Размер в Мегабайтах

# Оценка модели на тестовом наборе
accuracy, recall, precision, predict_time = evaluate_model(svm_clf, X_test, y_test)

# Выводим результаты
table = PrettyTable()
table.field_names = ["Параметр", "Значение"]

table.add_row(["Кол-во изображений для обучений", f"{total_images_count} шт"])
table.add_row(["Размер модели", f"{model_size:.2f} MB"])
table.add_row(["Точность", f"{accuracy:.4f}"])
table.add_row(["Полнота", f"{recall:.4f}"])
table.add_row(["Точность предсказаний", f"{precision:.4f}"])
table.add_row(["Время на предсказание", f"{predict_time * 1000:.2f} ms"])

print(table)

print("Модель успешно обучена и оценена")
