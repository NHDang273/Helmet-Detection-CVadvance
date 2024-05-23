import streamlit as st
import torch
from PIL import Image
import numpy as np
import cv2
from ultralytics import YOLO
# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client rider2
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vE1rnHyYmah1YSdCVEZY"
)

# infer on a local image
rider_result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="detect-rider2/3")
# import the inference-sdk
from inference_sdk import InferenceHTTPClient

# initialize the client
CLIENT = InferenceHTTPClient(
    api_url="https://detect.roboflow.com",
    api_key="vE1rnHyYmah1YSdCVEZY"
)

# infer on a local image
helmet_result = CLIENT.infer("YOUR_IMAGE.jpg", model_id="detect-helmet-ecdmd/1")







# Hàm phát hiện đối tượng
def detect_objects(model, image):
    results = model(image)
    return results

def main():
    st.title("Rider and Helmet Detection")

    

if __name__ == "__main__":
    main()
