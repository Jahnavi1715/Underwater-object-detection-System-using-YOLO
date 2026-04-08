import streamlit as st
import pandas as pd
import hashlib
from pathlib import Path
from PIL import Image
import tempfile
import cv2
from ultralytics import YOLO

USER_FILE = Path("users.xlsx")

# --- PASSWORD HASH FUNCTION ---
def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()

# --- LOAD USERS ---
def load_users():
    if USER_FILE.exists():
        return pd.read_excel(USER_FILE)
    else:
        return pd.DataFrame(columns=["email", "password"])

# --- SAVE USERS ---
def save_users(df):
    df.to_excel(USER_FILE, index=False)

# --- SIGNUP ---
def signup(email, password):
    users = load_users()
    if email in users["email"].values:
        return False, "Email already registered!"
    new_user = pd.DataFrame({"email": [email], "password": [hash_password(password)]})
    users = pd.concat([users, new_user], ignore_index=True)
    save_users(users)
    return True, "Sign-up successful! Please log in."

# --- LOGIN ---
def login(email, password):
    users = load_users()
    hashed = hash_password(password)
    match = users[(users["email"] == email) & (users["password"] == hashed)]
    return not match.empty


# ======================
# PAGE CONFIG
# ======================
st.set_page_config(
    page_title="Underwater Object Detection",
    page_icon="🐟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ======================
# SESSION INITIALIZATION
# ======================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

# ======================
# AUTHENTICATION UI
# ======================
st.sidebar.title("🔑 User Authentication")
mode = st.sidebar.radio("Choose mode:", ["Login", "Sign Up"])

if mode == "Sign Up":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Sign Up"):
        success, msg = signup(email, password)
        if success:
            st.sidebar.success(msg)
        else:
            st.sidebar.error(msg)

elif mode == "Login":
    email = st.sidebar.text_input("Email")
    password = st.sidebar.text_input("Password", type="password")
    if st.sidebar.button("Login"):
        if login(email, password):
            st.session_state["logged_in"] = True
            st.session_state["email"] = email
            st.sidebar.success("Login successful ✅")
        else:
            st.session_state["logged_in"] = False
            st.sidebar.error("Invalid email or password")

# ======================
# MAIN APP (Only if logged in)
# ======================
if st.session_state["logged_in"]:
    st.sidebar.write(f"👋 Welcome, {st.session_state['email']}!")

    st.title("🐟 Underwater Object Detection")
    st.markdown("Upload an image or video of underwater scenes and the model will detect fish or objects.")

    # ---- SELECT INPUT TYPE ---- #
    input_type = st.radio("Select input type:", ["Image", "Video"])

    # ---- LOAD MODELS ---- #
    @st.cache_resource
    def load_image_model():
        return YOLO("weights/yolov8n-cls1.pt")  # Classification

    @st.cache_resource
    def load_video_model():
        return YOLO("weights/yolov8n-cls2.pt")  # Detection

    model = load_image_model() if input_type == "Image" else load_video_model()
    st.sidebar.success(f"✅ {input_type} model loaded!")
    st.sidebar.subheader("Model Classes")
    st.sidebar.json(model.names)

    # ---- IMAGE MODE ---- #
    if input_type == "Image":
        uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
        if uploaded_image:
            img = Image.open(uploaded_image).convert("RGB")
            st.image(img, caption="Uploaded Image", use_container_width=True)

            if st.button("🔍 Detect in Image"):
                with st.spinner("Detecting..."):
                    results = model.predict(img)
                    result = results[0]
                    st.image(result.plot(), caption="Detected Image", use_container_width=True)

                    probs = result.probs
                    if probs is not None:
                        top5 = probs.top5
                        st.subheader("📊 Top-5 Predictions")
                        for i in range(5):
                            class_id = top5[i]
                            class_name = model.names[class_id]
                            confidence = probs.data[class_id].item()
                            st.write(f"{i+1}. {class_name}** — {confidence:.2%}")
                    else:
                        st.warning("No classification predictions found.")

    # ---- VIDEO MODE ---- #
    elif input_type == "Video":
        uploaded_video = st.file_uploader("Upload a video", type=["mp4", "avi", "mov"])
        if uploaded_video:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_video.read())
            video_path = tfile.name

            stframe = st.empty()  # For live frame display

            if st.button("🎥 Start Detection (Classification)"):
                with st.spinner("Analyzing video..."):
                    cap = cv2.VideoCapture(video_path)
                    if not cap.isOpened():
                        st.error("Error: Could not open video.")
                    else:
                        frame_count = 0
                        class_counts = {}

                        while cap.isOpened():
                            ret, frame = cap.read()
                            if not ret:
                                break

                            # Predict classification for the frame
                            results = model.predict(frame)
                            result = results[0]
                            annotated = result.plot()  # This adds predicted label text

                            # Update class counts
                            if result.probs is not None:
                                probs = result.probs
                                top_class_id = probs.top1
                                class_name = model.names[top_class_id]
                                class_counts[class_name] = class_counts.get(class_name, 0) + 1

                            # Display the annotated frame
                            annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
                            stframe.image(annotated, channels="RGB", use_container_width=True)

                            frame_count += 1
                            if frame_count % 3 != 0:
                                st.caption(f"Processed {frame_count} frames...")

                        cap.release()

                        st.success("✅ Classification completed!")
                        if class_counts:
                            st.subheader("📊 Most frequent classifications in video")
                            for cls, count in sorted(class_counts.items(), key=lambda x: x[1], reverse=True):
                                st.write(f"- *{cls}* — {count} frames")
                        else:
                            st.warning("No classifications detected in the video.")

else:
    st.warning("Please log in to access the app.")

