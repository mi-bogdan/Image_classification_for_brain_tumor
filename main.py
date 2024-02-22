from typing import Tuple
import numpy as np
import cv2
import streamlit as st
import joblib
from PIL import Image
from skimage.io import imread
from skimage.transform import resize

# Загрузка моделей


def load_model(kmeans_path: str = 'kmeans.joblib', svm_clf_path: str = 'svm_clf.joblib') -> Tuple:
    kmeans = joblib.load(kmeans_path)
    svm_clf = joblib.load(svm_clf_path)
    return kmeans, svm_clf

# Обработка изображения


def process_image(image_path, kmeans, n_clusters: int = 50) -> np.ndarray:
    image = imread(image_path, as_gray=True)
    image = resize(image, (128, 128), anti_aliasing=True)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute((image * 255).astype('uint8'), None)
    if des is not None:
        bovw_feature = kmeans.predict(des)
        bovw_histogram = np.bincount(
            bovw_feature, minlength=n_clusters).reshape(1, -1)
    else:
        bovw_histogram = np.zeros((1, n_clusters))
    return bovw_histogram

# Классификация изображения


def predict_image_class(image_path, kmeans, svm_clf) -> str:
    bovw_histogram = process_image(image_path, kmeans)
    prediction = svm_clf.predict(bovw_histogram)
    categories = ["no_tumor", "glioma_tumor",
                  "meningioma_tumor", "pituitary_tumor"]
    return categories[prediction[0]]

# Интерфейс Streamlit


def streamlit_interface():
    st.header('Загрузка и отображение изображения')
    uploaded_file = st.file_uploader(
        "Выберите файл изображения", type=['jpg', 'jpeg', 'png'])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        # Центрирование изображения с помощью колонок
        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write("")
        with col2:
            st.image(image, caption='Загруженное изображение.',
                     use_column_width=True)
        with col3:
            st.write("")
        col = st.columns([1])

        with col[0]:
            if st.button('Прогнозировать опухоль', key="full_width_button"):
                kmeans, svm_clf = load_model()
                predicted_class = predict_image_class(
                    uploaded_file, kmeans, svm_clf)
                if predicted_class == 'no_tumor':
                    st.success('Все хорошо, опухоли нет')
                elif predicted_class == 'glioma_tumor':
                    st.error('Опухоль глиомы')
                elif predicted_class == 'meningioma_tumor':
                    st.error('Опухоль менингиомы')
                else:
                    st.error('Опухоль гипофиза')

    else:
        st.warning('Пожалуйста, загрузите файл изображения.')


streamlit_interface()
