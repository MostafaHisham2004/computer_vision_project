import streamlit as st
import requests
from PIL import Image, ImageDraw
import io
import numpy as np

# API Endpoint
API_URL = "http://localhost:8000"

# Page configuration
st.set_page_config(
    page_title="Vision AI | Multi-Model Dashboard",
    page_icon="🤖",
    layout="wide"
)

# Custom CSS for Premium Look
st.markdown("""
    <style>
    .main {
        background-color: #0e1117;
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
        background-color: #ff4b4b;
        color: white;
        border: none;
        font-weight: bold;
    }
    .stButton>button:hover {
        background-color: #ff3333;
        border: none;
    }
    h1 {
        color: #f0f2f6;
        font-family: 'Inter', sans-serif;
        font-weight: 800;
    }
    .stats-card {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #ff4b4b;
        margin-bottom: 10px;
    }
    .sidebar .sidebar-content {
        background-image: linear-gradient(#2e3333, #1e2130);
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

st.title("🤖 Multi-Model Vision AI")
st.markdown("### Vehicle Detection | Segmentation | Classification")

# Sidebar
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2103/2103633.png", width=100)
st.sidebar.title("Configuration")

model_option = st.sidebar.selectbox(
    "Select AI Model",
    ("YOLO (Vehicle Detection)", "UNet (Semantic Segmentation)", "CNN (Plant Disease)")
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.25)
uploaded_file = st.sidebar.file_uploader("Upload Image Assets", type=["jpg", "jpeg", "png"])

st.sidebar.markdown("---")
st.sidebar.info("Model Backend: FastAPI @ " + API_URL)

# Main Content
col1, col2 = st.columns([2, 1])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    
    with col1:
        st.image(image, caption="Original Input", use_container_width=True)
        
        if st.button("🚀 Run AI Inference"):
            with st.spinner('Neural network is processing...'):
                # Prepare image for API
                img_byte_arr = io.BytesIO()
                image.save(img_byte_arr, format='JPEG')
                img_byte_arr = img_byte_arr.getvalue()
                files = {"file": ("image.jpg", img_byte_arr, "image/jpeg")}
                
                try:
                    if model_option == "YOLO (Vehicle Detection)":
                        response = requests.post(f"{API_URL}/predict/yolo", files=files)
                        if response.status_code == 200:
                            detections = response.json()["detections"]
                            # Filter detections
                            filtered = [d for d in detections if d["score"] >= confidence_threshold]
                            
                            draw = ImageDraw.Draw(image)
                            counts = {}
                            for det in filtered:
                                bbox = det["bbox"]
                                label = det["class"]
                                score = det["score"]
                                counts[label] = counts.get(label, 0) + 1
                                draw.rectangle(bbox, outline="#ff4b4b", width=4)
                                draw.text((bbox[0], bbox[1]-15), f"{label} {score:.2f}", fill="#ff4b4b")
                            
                            st.image(image, caption="Detection Results", use_container_width=True)
                            
                            with col2:
                                st.markdown("### 📊 Detection Stats")
                                if not filtered:
                                    st.warning("No objects found above threshold.")
                                else:
                                    for label, count in counts.items():
                                        st.markdown(f"""
                                        <div class="stats-card">
                                            <h4 style='margin:0; color:#bdc3c7;'>{label.upper()}</h4>
                                            <h2 style='margin:0; color:white;'>{count}</h2>
                                        </div>
                                        """, unsafe_allow_html=True)

                    elif model_option == "UNet (Semantic Segmentation)":
                        response = requests.post(f"{API_URL}/predict/unet", files=files)
                        if response.status_code == 200:
                            mask = np.array(response.json()["mask"], dtype=np.uint8)
                            colors = np.array([
                                [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
                                [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
                                [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
                                [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
                            ])
                            colored_mask = colors[mask]
                            mask_image = Image.fromarray(colored_mask.astype(np.uint8))
                            mask_image = mask_image.resize(image.size, resample=Image.NEAREST)
                            overlay = Image.blend(image, mask_image, alpha=0.5)
                            st.image(overlay, caption="Segmentation Overlay", use_container_width=True)
                            
                            with col2:
                                st.markdown("### 🗺️ Semantic Map")
                                st.image(mask_image, use_container_width=True)
                                st.info("Colors represent different Cityscapes categories (Road, Sidewalk, Sky, etc.)")

                    elif model_option == "CNN (Plant Disease)":
                        response = requests.post(f"{API_URL}/predict/cnn", files=files)
                        if response.status_code == 200:
                            res = response.json()
                            with col2:
                                st.markdown("### 🍃 Diagnosis")
                                st.markdown(f"""
                                <div class="stats-card" style="border-left: 5px solid #2ecc71;">
                                    <h4 style='margin:0; color:#bdc3c7;'>PREDICTED CLASS</h4>
                                    <h3 style='margin:0; color:white;'>{res['label'].replace('___', ' - ')}</h3>
                                </div>
                                <div class="stats-card" style="border-left: 5px solid #3498db;">
                                    <h4 style='margin:0; color:#bdc3c7;'>CONFIDENCE</h4>
                                    <h2 style='margin:0; color:white;'>{res['confidence']:.2%}</h2>
                                </div>
                                """, unsafe_allow_html=True)
                                if "healthy" in res['label'].lower():
                                    st.success("The plant appears to be healthy!")
                                else:
                                    st.error("Action required: Disease detected.")
                        
                    else:
                        st.error(f"Error: API returned status {response.status_code}")
                except Exception as e:
                    st.error(f"Connection Error: {e}")

else:
    with col1:
        st.info("Please upload an image via the sidebar to begin analysis.")
    with col2:
        st.markdown("### ℹ️ Operations")
        st.write("""
        1. **Select Model**: Choose between object detection, pixel-level segmentation, or disease classification.
        2. **Threshold**: Adjust sensitivity for detection results.
        3. **Analyze**: AI processes the image and returns visual & statistical feedback.
        """)

st.markdown("---")
st.markdown("<p style='text-align: center; color: #bdc3c7;'>Integrated Multi-Model AI Deployment System</p>", unsafe_allow_html=True)
