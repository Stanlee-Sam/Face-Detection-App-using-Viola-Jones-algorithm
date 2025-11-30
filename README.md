# 👤 Face Detection App using Viola-Jones Algorithm

## Overview

This is a **Streamlit web application** for **face detection** using the **Viola-Jones algorithm (Haar Cascade)**. The app can detect faces in **real-time from a webcam** or from **uploaded images**. Users can customize detection parameters, choose rectangle colors, and optionally save processed images.

---

## Features

* Real-time face detection from **webcam**.
* Face detection from **uploaded images** (JPG, JPEG, PNG).
* **Custom rectangle color** using a color picker.
* Adjustable **scaleFactor** and **minNeighbors** parameters for fine-tuning detection.
* Option to **save images** with detected faces.
* Displays **number of faces detected** in real-time.
* User-friendly interface with **instructions and tips**.

---

## Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/face-detection-app.git
cd face-detection-app
```

2. **Create a virtual environment (optional but recommended):**

```bash
python -m venv venv
# Activate it
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

3. **Install dependencies:**

```bash
pip install -r requirements.txt
```

**If you don’t have a `requirements.txt`, install manually:**

```bash
pip install streamlit opencv-python numpy
```

---

## Usage

1. **Run the Streamlit app:**

```bash
streamlit run app.py
```

2. **Open your browser** (Streamlit usually opens automatically).

3. **Instructions in the app:**

   * **Webcam Detection**:

     * Click **Start Detection** to begin.
     * Use **Stop Detection** to stop.
     * If saving is enabled, click **Capture Frame** to save the current frame.
   * **Image Upload Detection**:

     * Upload an image (JPG, JPEG, PNG).
     * Faces will be detected automatically.
     * Images can be saved if the option is enabled.

4. **Customize detection parameters** in the sidebar:

   * **Scale Factor**: Controls detection accuracy vs speed (1.1 = more accurate but slower).
   * **Min Neighbors**: Higher = fewer false positives.
   * **Rectangle Color**: Pick your preferred rectangle color.
   * **Save Images**: Toggle to save detected images.

---

## Tips for Better Detection

* Ensure **good lighting**.
* Face the camera directly.
* Maintain a reasonable **distance from the camera**.
* Adjust **scaleFactor** and **minNeighbors** if detection is too sensitive or missing faces.

---

## Troubleshooting

* **No faces detected**: Lower the `Min Neighbors` value.
* **Too many false positives**: Increase the `Min Neighbors` value.
* **Slow detection**: Increase the `Scale Factor`.

---

## Dependencies

* Python 3.7+
* [Streamlit](https://streamlit.io/)
* [OpenCV](https://opencv.org/)
* Numpy

---

## Screenshots

**Webcam Detection:**
![Webcam Detection](screenshots/webcam_detection.png)

**Image Upload Detection:**
![Image Detection](screenshots/image_detection.png)

---

## License

This project is **open-source** under the **MIT License**.

---

