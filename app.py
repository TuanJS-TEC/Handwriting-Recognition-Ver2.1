# app.py
import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import os
from streamlit_drawable_canvas import st_canvas

# ----------------------
# Load mô hình
# ----------------------
@st.cache_resource
def load_models():
    mnist = load_model("best_mnist.h5", compile=False)
    shape = load_model("best_shapes.h5", compile=False)
    return mnist, shape

mnist_model, shape_model = load_models()

# ----------------------
# Hằng số
# ----------------------
MNIST_IMG_SIZE = 28
SHAPE_IMG_SIZE = 64
SHAPE_CLASSES = ["circle", "square", "triangle"]

# ----------------------
# Hàm helper
# ----------------------
def preprocess_image(img: Image.Image, size: int) -> np.ndarray:
    img = img.convert("L").resize((size, size))
    arr = np.array(img) / 255.0
    return arr.reshape(1, size, size, 1)

def predict_mnist(img: Image.Image):
    arr = preprocess_image(img, MNIST_IMG_SIZE)
    pred = mnist_model.predict(arr)
    return np.argmax(pred), pred[0]

def predict_shape(img: Image.Image):
    arr = preprocess_image(img, SHAPE_IMG_SIZE)
    pred = shape_model.predict(arr)
    cls_idx = np.argmax(pred)
    return SHAPE_CLASSES[cls_idx], pred[0], SHAPE_CLASSES

def save_image(img: Image.Image, prefix: str) -> str:
    os.makedirs("savepic", exist_ok=True)
    path = os.path.join("savepic", f"{prefix}_{np.random.randint(1000,9999)}.png")
    img.save(path)
    return path

def handle_canvas(canvas_result, predict_func, prefix):
    if canvas_result.image_data is not None:
        img_array = canvas_result.image_data
        img = Image.fromarray((img_array[:, :, 0]).astype(np.uint8))
        result = predict_func(img)
        save_path = save_image(img, prefix)
        return result, save_path
    return None, None

# ----------------------
# Streamlit UI
# ----------------------
st.set_page_config(page_title="MNIST & Shapes Predictor", layout="centered")
st.title("🖌️ MNIST & Geometric Shapes Predictor")

# Sidebar để chọn trang
page = st.sidebar.selectbox("Chọn trang", ["Dự đoán MNIST", "Dự đoán Hình học"])

# ----------------------
# Trang MNIST
# ----------------------
if page == "Dự đoán MNIST":
    st.header("📄 Dự đoán chữ số MNIST")
    option = st.radio("Chọn kiểu input:", ["Upload ảnh", "Vẽ tay"])

    if option == "Upload ảnh":
        uploaded = st.file_uploader("Chọn ảnh MNIST", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            # **Đã thay** use_column_width -> use_container_width
            st.image(img, caption="Ảnh đầu vào", use_container_width=True)
            pred_class, pred_probs = predict_mnist(img)
            st.subheader(f"Dự đoán: {pred_class}")
            st.write("Xác suất từng lớp:")
            for i, p in enumerate(pred_probs):
                st.write(f"{i}: {p*100:.2f}%")

    else:  # Vẽ tay
        st.write("Vẽ chữ số bằng chuột trái:")
        canvas_result = st_canvas(
            fill_color="white",
            stroke_width=10,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="mnist_canvas",
        )

        if st.button("Dự đoán & Lưu MNIST"):
            (pred_class, pred_probs), save_path = handle_canvas(canvas_result, predict_mnist, "mnist")
            if pred_class:
                st.subheader(f"Dự đoán: {pred_class}")
                st.write("Xác suất từng lớp:")
                for i, p in enumerate(pred_probs):
                    st.write(f"{i}: {p*100:.2f}%")
                st.success(f"Ảnh MNIST đã lưu tại: {save_path}")

# ----------------------
# Trang Hình học
# ----------------------
else:
    st.header("🔺 Dự đoán hình học")
    option = st.radio("Chọn kiểu input:", ["Upload ảnh", "Vẽ tay"])

    if option == "Upload ảnh":
        uploaded = st.file_uploader("Chọn ảnh hình học", type=["png", "jpg", "jpeg"])
        if uploaded:
            img = Image.open(uploaded)
            # **Đã thay** use_column_width -> use_container_width
            st.image(img, caption="Ảnh đầu vào", use_container_width=True)
            pred_class, pred_probs, classes = predict_shape(img)
            st.subheader(f"Dự đoán: {pred_class}")
            st.write("Xác suất từng lớp:")
            for cls, p in zip(classes, pred_probs):
                st.write(f"{cls}: {p*100:.2f}%")

    else:  # Vẽ tay
        st.write("Vẽ hình bằng chuột trái:")
        canvas_result = st_canvas(
            fill_color="black",
            stroke_width=5,
            stroke_color="white",
            background_color="black",
            height=280,
            width=280,
            drawing_mode="freedraw",
            key="shape_canvas",
        )

        if st.button("Dự đoán & Lưu Hình"):
            (pred_class, pred_probs, classes), save_path = handle_canvas(canvas_result, predict_shape, "shape")
            if pred_class:
                st.subheader(f"Dự đoán: {pred_class}")
                st.write("Xác suất từng lớp:")
                for cls, p in zip(classes, pred_probs):
                    st.write(f"{cls}: {p*100:.2f}%")
                st.success(f"Ảnh đã lưu tại: {save_path}")
