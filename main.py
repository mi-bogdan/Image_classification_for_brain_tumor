from typing import Tuple
import numpy as np
import cv2
import streamlit as st
import joblib
from PIL import Image
from skimage.io import imread
from skimage.transform import resize


def load_model(svm_clf_path: str = 'svm_clf.joblib'):
    """Загрузка модели SVM"""
    svm_clf = joblib.load(svm_clf_path)
    return svm_clf

def process_image(image_path) -> np.ndarray:
    """Обработка изображения и извлечение агрегированных дескрипторов SIFT"""
    image = imread(image_path, as_gray=True)
    image = resize(image, (128, 128), anti_aliasing=True)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute((image * 255).astype('uint8'), None)
    if des is not None:
        agg_descriptors = np.mean(des, axis=0).reshape(1, -1)  # Агрегируем, используя среднее значение
    else:
        agg_descriptors = np.zeros((1, 128))  # Предполагаем, что дескриптор SIFT имеет размер 128
    return agg_descriptors

def predict_image_class(image_path, svm_clf) -> str:
    """Классификация изображения"""
    agg_descriptors = process_image(image_path)
    prediction = svm_clf.predict(agg_descriptors)
    categories = ["no_tumor", "glioma_tumor", "meningioma_tumor", "pituitary_tumor"]
    return categories[prediction[0]]

def streamlit_interface():
    """Интерфейс Streamlit"""
    st.header('Распознавание опухолей головного мозга')
    uploaded_file = st.file_uploader("Выберите файл изображения", type=['jpg', 'jpeg', 'png'])

    dict_classes = {
        'no_tumor': 'Все хорошо, опухоли нет!',
        'glioma_tumor': 'Опухоль глиомы!',
        'meningioma_tumor': 'Опухоль менингиомы!',
        'pituitary_tumor': 'Опухоль гипофиза!'
    }

    if uploaded_file is not None:
        image = Image.open(uploaded_file)

        col1, col2, col3 = st.columns([1, 2, 1])
        with col1:
            st.write("")
        with col2:
            st.image(image, caption='Загруженное изображение.', use_column_width=True)
        with col3:
            st.write("")

        if st.button('Прогнозировать опухоль', key="predict"):
            svm_clf = load_model('svm_clf.joblib')
            with st.spinner('Анализ изображения...'):
                predicted_class = predict_image_class(uploaded_file, svm_clf)
            if predicted_class == 'no_tumor':
                st.success(dict_classes[predicted_class])
            else:
                st.error(dict_classes[predicted_class])
    else:
        st.warning('Пожалуйста, загрузите файл изображения.')

streamlit_interface()
