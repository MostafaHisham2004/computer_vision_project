import torch
import torchvision.transforms as T
from torchvision.models.segmentation import deeplabv3_resnet50
from ultralytics import YOLO
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import os

class YoloService:
    def __init__(self, model_path):
        self.model = YOLO(model_path)
        self.class_names = ['car', 'threewheel', 'bus', 'truck', 'motorbike', 'van']

    def predict(self, image):
        results = self.model.predict(image, imgsz=640)
        return results[0]

class SegmentationService:
    def __init__(self, model_path):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.num_classes = 19
        
        try:
            # Initialize model architecture
            # Note: We initialize with aux_loss=False (default) for inference efficiency.
            # If the checkpoint contains aux_classifier weights, we filter them out.
            self.model = deeplabv3_resnet50(num_classes=self.num_classes, aux_loss=False)
            
            print(f"Loading UNet model from {model_path}...")
            state_dict = torch.load(model_path, map_location=self.device)
            
            # Identify keys in checkpoint that are not in the current model architecture
            model_keys = self.model.state_dict().keys()
            checkpoint_keys = state_dict.keys()
            
            unexpected_keys = [k for k in checkpoint_keys if k not in model_keys]
            missing_keys = [k for k in model_keys if k not in checkpoint_keys]
            
            if unexpected_keys:
                print(f"WARNING: Unexpected keys in checkpoint: {unexpected_keys}. These will be ignored.")
                # Filter state_dict to only include keys present in the model
                state_dict = {k: v for k, v in state_dict.items() if k in model_keys}
            
            if missing_keys:
                raise RuntimeError(f"Missing keys in checkpoint: {missing_keys}")
                
            self.model.load_state_dict(state_dict, strict=True)
            self.model.to(self.device)
            self.model.eval()
            print("UNet model loaded successfully.")
            
        except Exception as e:
            print(f"CRITICAL: Failed to load UNet model: {str(e)}")
            raise

        self.transform = T.Compose([
            T.Resize((256, 512)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.class_names = [
            'road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light',
            'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car',
            'truck', 'bus', 'train', 'motorcycle', 'bicycle'
        ]
        
        # Color map for Cityscapes
        self.colors = np.array([
            [128, 64, 128], [244, 35, 232], [70, 70, 70], [102, 102, 156], [190, 153, 153],
            [153, 153, 153], [250, 170, 30], [220, 220, 0], [107, 142, 35], [152, 251, 152],
            [70, 130, 180], [220, 20, 60], [255, 0, 0], [0, 0, 142], [0, 0, 70],
            [0, 60, 100], [0, 80, 100], [0, 0, 230], [119, 11, 32]
        ], dtype=np.uint8)

    def predict(self, image_pil):
        input_tensor = self.transform(image_pil).unsqueeze(0).to(self.device)
        with torch.no_grad():
            # DeepLabV3 returns a dict; we want the 'out' key
            output = self.model(input_tensor)['out'][0]
        output_predictions = output.argmax(0).cpu().numpy()
        return output_predictions

class ClassificationService:
    def __init__(self, model_path):
        # Configure TensorFlow to only allocate necessary memory
        # This prevents it from clashing with PyTorch/YOLO/UNet
        gpus = tf.config.list_physical_devices('GPU')
        if gpus:
            try:
                for gpu in gpus:
                    tf.config.experimental.set_memory_growth(gpu, True)
                print("TensorFlow: GPU memory growth enabled.")
            except RuntimeError as e:
                print(f"TensorFlow GPU config error: {e}")

        try:
            print(f"Loading CNN model from {model_path}...")
            self.model = tf.keras.models.load_model(model_path, compile=False)
            print("CNN model loaded successfully.")
        except Exception as e:
            print(f"Standard load_model failed: {e}")
            print("Attempting to reconstruct model architecture and load weights...")
            try:
                # Reconstruct the exact architecture from the notebook
                base_model = tf.keras.applications.MobileNetV2(
                    weights=None,
                    include_top=False,
                    input_shape=(128, 128, 3)
                )
                self.model = tf.keras.models.Sequential([
                    base_model,
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(256, activation='relu'),
                    tf.keras.layers.Dropout(0.5),
                    tf.keras.layers.Dense(38, activation='softmax')
                ])
                # Load weights from the H5 file (it contains both weights and arch)
                self.model.load_weights(model_path)
                print("CNN model reconstructed and weights loaded successfully.")
            except Exception as e2:
                print(f"Reconstruction failed: {e2}")
                # Last resort fallback to CPU if not already tried
                print("Attempting to load CNN model on CPU as a last resort...")
                with tf.device('/CPU:0'):
                    try:
                        self.model = tf.keras.models.load_model(model_path, compile=False)
                        print("CNN model loaded successfully on CPU.")
                    except Exception as e3:
                        print(f"Critical failure loading CNN model: {e3}")
                        raise e3

        self.img_size = (128, 128)
        self.class_names = [
            'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
            'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy',
            'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_',
            'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot',
            'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)', 'Grape___healthy',
            'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot', 'Peach___healthy',
            'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight',
            'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
            'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
            'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
            'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
            'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
        ]

    def predict(self, image_pil):
        img = image_pil.resize(self.img_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predictions = self.model.predict(img_array)
        score = tf.nn.softmax(predictions[0])
        class_idx = np.argmax(predictions[0])
        return self.class_names[class_idx], float(predictions[0][class_idx])
