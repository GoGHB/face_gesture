import csv
import os
import threading
import time
from collections import deque
from datetime import datetime

import av
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from PIL import Image
from streamlit_webrtc import VideoProcessorBase, WebRtcMode, webrtc_streamer

st.set_page_config(
    page_title="Facial Emotion Recognition AI",
    page_icon="🎭",
    layout="wide",
)

MODEL_PATH = "facial_emotion_rescnn.keras"
RATINGS_FILE = "ratings.csv"

CLASS_NAMES = ["angry", "disgust", "fear", "happy", "neutral", "sad", "surprise"]
EMOJI = {
    "angry": "😠", "disgust": "🤢", "fear": "😨", "happy": "😄",
    "neutral": "😐", "sad": "😢", "surprise": "😲",
}

MAX_POINTS = 300     # readings kept in memory
PLOT_POINTS = 60     # readings shown on the live graph
RECENT_POINTS = 10   # readings used for "recent mood"

RTC_CONFIG = {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}


# ------------------------------------------------------------------
# Model + detector
# ------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading emotion model...")
def load_model():
    return tf.keras.models.load_model(MODEL_PATH, compile=False)


def make_detector():
    if not hasattr(cv2, "CascadeClassifier"):
        st.error('OpenCV 5 detected. Use "opencv-python-headless<5" in requirements.txt.')
        st.stop()
    return cv2.CascadeClassifier(
        cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    )


model = load_model()
main_detector = make_detector()


def predict_faces(mdl, detector, gray):
    """Returns a list of ((x, y, w, h), probabilities) for every face."""
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.2, minNeighbors=5, minSize=(60, 60)
    )
    results = []
    for (x, y, w, h) in faces:
        face = cv2.resize(gray[y:y + h, x:x + w], (48, 48)).astype(np.float32)
        face = np.expand_dims(face, axis=(0, -1))  # same preprocessing as training app
        preds = mdl(face, training=False).numpy()[0]
        results.append(((int(x), int(y), int(w), int(h)), preds))
    return results


def draw_results(img, results, color=(0, 200, 0)):
    for (x, y, w, h), preds in results:
        idx = int(np.argmax(preds))
        label = f"{CLASS_NAMES[idx]} {preds[idx] * 100:.0f}%"
        cv2.rectangle(img, (x, y), (x + w, y + h), color, 3)
        cv2.putText(
            img, label, (x, max(y - 10, 25)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2,
        )
    return img


# ------------------------------------------------------------------
# Live video processor (runs in a background thread; nothing is saved)
# ------------------------------------------------------------------
class EmotionProcessor(VideoProcessorBase):
    def __init__(self, mdl):
        self.model = mdl
        self.detector = make_detector()
        self.lock = threading.Lock()
        self.frame_no = 0
        self.last_results = []
        self.fps = 0.0
        self.last_t = None
        self.reset()

    def reset(self):
        with self.lock:
            self.t0 = time.time()
            self.history = deque(maxlen=MAX_POINTS)
            self.counts = {c: 0 for c in CLASS_NAMES}
            self.n = 0

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        self.frame_no += 1

        # Analyse every 3rd frame to keep the video smooth on CPU
        if self.frame_no % 3 == 1:
            now = time.time()
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            results = predict_faces(self.model, self.detector, gray)
            self.last_results = results

            if self.last_t is not None:
                dt = now - self.last_t
                if dt > 0:
                    inst = 1.0 / dt
                    self.fps = inst if not self.fps else 0.8 * self.fps + 0.2 * inst
            self.last_t = now

            if results:
                biggest = max(results, key=lambda r: r[0][2] * r[0][3])
                scores = biggest[1]
                with self.lock:
                    self.history.append((now - self.t0, scores.copy()))
                    self.counts[CLASS_NAMES[int(np.argmax(scores))]] += 1
                    self.n += 1

        draw_results(img, self.last_results)
        return av.VideoFrame.from_ndarray(img, format="bgr24")

    def snapshot(self):
        with self.lock:
            return list(self.history), dict(self.counts), self.n, self.fps


def render_dashboard(proc, timeline_ph, mix_ph, insights_ph):
    hist, counts, n, fps = proc.snapshot()

    if not hist:
        insights_ph.info("Waiting for a face... look at the camera.")
        return

    recent = hist[-PLOT_POINTS:]
    df = pd.DataFrame(
        [s * 100 for _, s in recent],
        columns=CLASS_NAMES,
    )
    timeline_ph.line_chart(df, height=300)

    mix = pd.Series({c: counts[c] / n * 100 for c in CLASS_NAMES}, name="% of session")
    mix_ph.bar_chart(mix, height=220)

    last = hist[-RECENT_POINTS:]
    avg = np.mean([s for _, s in last], axis=0)
    stable = CLASS_NAMES[int(np.argmax(avg))]
    overall = max(counts, key=counts.get)
    insights_ph.markdown(
        f"**Recent mood:** {EMOJI[stable]} {stable.upper()} ({avg.max() * 100:.0f}%)  \n"
        f"**Most frequent this session:** {EMOJI[overall]} {overall.upper()}  \n"
        f"**Readings analysed:** {n}  \n"
        f"**Analysis speed:** {fps:.1f} per second"
    )


# ------------------------------------------------------------------
# Single image analysis (snapshot + upload tabs)
# ------------------------------------------------------------------
def analyze_image(pil_image):
    rgb = np.array(pil_image.convert("RGB"))
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    results = predict_faces(model, main_detector, gray)

    if not results:
        st.image(rgb, use_container_width=True)
        st.warning("No face detected. Face the camera in good light.")
        return

    annotated = draw_results(rgb.copy(), results)
    col1, col2 = st.columns(2)
    with col1:
        st.image(annotated, caption="Detected emotion", use_container_width=True)
    with col2:
        biggest = max(results, key=lambda r: r[0][2] * r[0][3])
        preds = biggest[1]
        idx = int(np.argmax(preds))
        st.subheader(f"{EMOJI[CLASS_NAMES[idx]]} {CLASS_NAMES[idx].upper()}")
        st.metric("Confidence", f"{preds[idx] * 100:.1f}%")
        st.bar_chart(
            pd.Series({c: float(preds[i]) * 100 for i, c in enumerate(CLASS_NAMES)}),
            height=250,
        )
        if len(results) > 1:
            st.caption(f"{len(results)} faces found. Showing the largest one.")


# ------------------------------------------------------------------
# Star rating (sidebar)
# ------------------------------------------------------------------
_lock = threading.Lock()


def read_ratings():
    if not os.path.exists(RATINGS_FILE):
        return []
    with open(RATINGS_FILE, newline="", encoding="utf-8") as f:
        return [int(r["stars"]) for r in csv.DictReader(f) if r.get("stars", "").isdigit()]


def rating_summary():
    stars = read_ratings()
    if not stars:
        return "No ratings yet. Be the first to rate!"
    avg = sum(stars) / len(stars)
    full = int(round(avg))
    return f"{'⭐' * full}{'☆' * (5 - full)}  {avg:.1f} / 5  ({len(stars)} ratings)"


def save_rating(stars, comment):
    with _lock:
        new_file = not os.path.exists(RATINGS_FILE)
        with open(RATINGS_FILE, "a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new_file:
                w.writerow(["time", "stars", "comment"])
            w.writerow([datetime.now().isoformat(timespec="seconds"), stars, comment.strip()])


with st.sidebar:
    st.header("⭐ Rate this app")
    selected = st.feedback("stars", key="star_rating")  # 0-4 or None
    comment = st.text_area("Comment (optional)", height=80, key="rating_comment")
    if st.button("Submit rating", type="primary"):
        if selected is None:
            st.warning("Please select a star rating first.")
        else:
            save_rating(selected + 1, comment)
            st.success("Thank you for your feedback! 🙏")
    st.caption(rating_summary())
    st.divider()
    st.caption(
        "Video is analysed frame by frame in memory. Nothing is recorded or saved. "
        "Predictions are estimates, not reliable measurements of a person's actual feelings."
    )


# ------------------------------------------------------------------
# Main page
# ------------------------------------------------------------------
st.title("🎭 Facial Emotion Recognition AI")
st.write(
    "Residual CNN based emotion recognition. Detects: Angry, Disgust, Fear, "
    "Happy, Neutral, Sad and Surprise."
)

tab_live, tab_snap, tab_upload = st.tabs(
    ["📷 Live Camera", "📸 Camera Snapshot", "🖼️ Upload Image"]
)

with tab_live:
    st.caption(
        "Click START and allow camera access. If the video does not connect on this "
        "network, use the Camera Snapshot tab instead."
    )
    ctx = webrtc_streamer(
        key="emotion-live",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={"video": True, "audio": False},
        video_processor_factory=lambda: EmotionProcessor(model),
        async_processing=True,
    )

    st.markdown("### 📊 Live Analysis")
    left, right = st.columns([2, 1])
    with left:
        st.caption("Emotion timeline (confidence %, last ~60 readings)")
        timeline_ph = st.empty()
    with right:
        insights_ph = st.empty()
        st.caption("Session emotion mix (%)")
        mix_ph = st.empty()
        if st.button("🔄 Reset session"):
            if ctx.video_processor:
                ctx.video_processor.reset()

with tab_snap:
    shot = st.camera_input("Take a photo")
    if shot is not None:
        analyze_image(Image.open(shot))

with tab_upload:
    uploaded = st.file_uploader("Upload a face image", type=["jpg", "jpeg", "png", "webp"])
    if uploaded is not None:
        analyze_image(Image.open(uploaded))

# The live loop must stay LAST: it keeps the script running while the camera is on.
if ctx.state.playing:
    while ctx.state.playing:
        if ctx.video_processor:
            render_dashboard(ctx.video_processor, timeline_ph, mix_ph, insights_ph)
        time.sleep(0.5)
else:
    insights_ph.info("Start the camera to see live analysis.")