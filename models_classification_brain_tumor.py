import cv2
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, recall_score, precision_score
from skimage.io import imread
from skimage.transform import resize
from tqdm import tqdm
import os
import time

os.environ['LOKY_MAX_CPU_COUNT'] = '4'


def load_images_and_labels(base_dir="Training"):
    categories = ["no_tumor", "glioma_tumor",
                  "meningioma_tumor", "pituitary_tumor"]
    images = []
    labels = []
    count = 0  # счетчик для подсчета количества изображений

    for label, category in enumerate(categories):
        path = os.path.join(base_dir, category)
        for img_name in tqdm(os.listdir(path), desc=f"Загрузка изображений из {category}"):
            img_path = os.path.join(path, img_name)
            img = imread(img_path, as_gray=True)
            img = resize(img, (128, 128), anti_aliasing=True)

            images.append(img)
            labels.append(label)
            count += 1

    return np.array(images), np.array(labels), count


def extract_sift_descriptors(images):
    sift = cv2.SIFT_create()
    descriptors_list = []

    for img in tqdm(images, desc="Извлечение дескрипторов SIFT"):
        kp, des = sift.detectAndCompute((img*255).astype('uint8'), None)
        if des is not None:
            descriptors_list.append(des)

    return descriptors_list


def build_bovw_features(descriptors_list, n_clusters=50):
    kmeans = KMeans(n_clusters=n_clusters)
    all_descriptors = np.vstack(descriptors_list)
    kmeans.fit(all_descriptors)
    bovw_features = [kmeans.predict(des) for des in descriptors_list]
    bovw_histograms = np.array(
        [np.bincount(feature, minlength=n_clusters) for feature in bovw_features])

    return bovw_histograms, kmeans


def train_svm_classifier(features, labels):
    svm_clf = make_pipeline(StandardScaler(), SVC(kernel='linear'))
    svm_clf.fit(features, labels)
    return svm_clf


def evaluate_model(model, X_test, y_test):
    start_time = time.time()
    predictions = model.predict(X_test)
    predict_time = time.time() - start_time

    accuracy = accuracy_score(y_test, predictions)
    recall = recall_score(y_test, predictions, average='weighted')
    precision = precision_score(y_test, predictions, average='weighted')

    return accuracy, recall, precision, predict_time


# Загружаем изображения и метки
images, labels, total_images_count = load_images_and_labels()

# Извлекаем дескрипторы SIFT
descriptors_list = extract_sift_descriptors(images)

# Строим признаки на основе Bag of Visual Words
bovw_features, kmeans = build_bovw_features(descriptors_list)

# Разделяем данные на обучающую и тестовую выборки
X_train, X_test, y_train, y_test = train_test_split(
    bovw_features, labels, test_size=0.3, random_state=42)

# Обучаем классификатор SVM
svm_clf = train_svm_classifier(X_train, y_train)

# Сохраняем модель и объект KMeans
joblib.dump(svm_clf, 'svm_clf.joblib')
joblib.dump(kmeans, 'kmeans.joblib')
model_size = os.path.getsize('svm_clf.joblib') / \
    (1024 * 1024)  # Размер в Мегабайтах

# Оцениваем модель на тестовом наборе
accuracy, recall, precision, predict_time = evaluate_model(
    svm_clf, X_test, y_test)

# Выводим результаты
print(f"Кол-во изображений для обучений: {total_images_count} шт")
print(f"Размер модели: {model_size:.2f} MB")
print(
    f"Точность: {accuracy:.4f}, Полнота: {recall:.4f}, Точность предсказаний: {precision:.4f}")
print(f"Время на предсказание: {predict_time * 1000:.2f} ms")

print("Модель успешно обучена и оценена")
