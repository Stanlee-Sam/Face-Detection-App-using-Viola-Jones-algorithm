import streamlit as st
import cv2
import numpy as np
from datetime import datetime

# Load Haar cascade classifier
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def hex_to_bgr(hex_color):
    """Convert hex color to BGR for OpenCV"""
    hex_color = hex_color.lstrip('#')
    rgb = tuple(int(hex_color[i:i+2], 16) for i in (0, 2, 4))
    return (rgb[2], rgb[1], rgb[0])  # Convert to BGR

def detect_faces_from_webcam(rect_color, min_neighbors, scale_factor, save_images):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("❌ Could not open webcam. Check permissions.")
        return

    stframe = st.empty()
    faces_count = st.empty()
    stop = st.button("🛑 Stop Detection")

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("❌ Could not read frame from webcam.")
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))

        bgr_color = hex_to_bgr(rect_color)
        for i, (x, y, w, h) in enumerate(faces, start=1):
            cv2.rectangle(frame, (x, y), (x+w, y+h), bgr_color, 2)
            cv2.putText(frame, f"Face {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, bgr_color, 1)

        faces_count.metric("👤 Faces Detected", len(faces))
        stframe.image(frame, channels="BGR", use_column_width=True)

        if save_images and st.button("📸 Capture Frame"):
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_webcam_{timestamp}.jpg"
            cv2.imwrite(filename, frame)
            st.success(f"Saved image as {filename}")

        if stop:
            break

    cap.release()
    st.info("📹 Webcam released successfully")

def detect_faces_from_image(uploaded_file, rect_color, min_neighbors, scale_factor, save_images):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(30, 30))
    bgr_color = hex_to_bgr(rect_color)
    for i, (x, y, w, h) in enumerate(faces, start=1):
        cv2.rectangle(image, (x, y), (x+w, y+h), bgr_color, 2)
        cv2.putText(image, f"Face {i}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr_color, 2)

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📊 Detection Results")
        st.metric("👤 Faces Detected", len(faces))
        if save_images and len(faces) > 0:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"face_image_{timestamp}.jpg"
            cv2.imwrite(filename, image)
            st.success(f"Saved image as {filename}")
    with col2:
        st.subheader("🖼️ Processed Image")
        st.image(image, channels="BGR", use_column_width=True)

def app():
    st.set_page_config(page_title="Face Detection App", page_icon="👤", layout="wide")
    st.title("👤 Face Detection using Viola-Jones Algorithm")

    st.markdown("""
    ### 📋 Instructions
    **Welcome!** This app uses the Viola-Jones algorithm to detect faces.

    **How to Use:**
    1. Customize detection with sliders & color picker in the sidebar
    2. Choose input source: Webcam or Image upload
    3. Enable saving to store processed images
    4. Adjust parameters for better detection

    **Parameter Guide:**
    - **Scale Factor**: Controls detection accuracy vs speed (1.1 = slower but better)
    - **Min Neighbors**: Higher = fewer false positives
    - **Rectangle Color**: Pick your preferred color
    """)

    # Sidebar controls
    st.sidebar.header("🎛️ Detection Controls")
    rect_color = st.sidebar.color_picker("🎨 Rectangle Color", "#00FF00")
    min_neighbors = st.sidebar.slider("👥 Min Neighbors", 1, 10, 5)
    scale_factor = st.sidebar.slider("🔍 Scale Factor", 1.1, 2.0, 1.3, 0.1)
    save_images = st.sidebar.checkbox("💾 Save Images", value=False)

    # Input source
    st.sidebar.header("📥 Input Source")
    input_source = st.sidebar.radio("Choose input source:", ["📹 Webcam", "🖼️ Upload Image"])

    if input_source == "📹 Webcam":
        st.header("📹 Real-time Face Detection")
        if st.button("🚀 Start Detection"):
            detect_faces_from_webcam(rect_color, min_neighbors, scale_factor, save_images)
    else:
        st.header("🖼️ Image Face Detection")
        uploaded_file = st.file_uploader("📤 Upload an image", type=['jpg', 'jpeg', 'png'])
        if uploaded_file:
            detect_faces_from_image(uploaded_file, rect_color, min_neighbors, scale_factor, save_images)

if __name__ == "__main__":
    app()
