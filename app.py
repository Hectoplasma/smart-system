import gradio as gr
from ultralytics import YOLO
import cv2
import numpy as np
from PIL import Image

# Load the YOLO model
model = YOLO("best.pt")  # Replace with the path to your trained model

# Function to process uploaded images and perform detection
def detect_objects(image):
    # Convert PIL image to OpenCV format
    image = np.array(image)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    # Perform object detection
    results = model(image)
    annotated_image = results[0].plot()  # Annotate the image with bounding boxes and labels
    
    # Convert the annotated image back to PIL format for display
    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(annotated_image)

# Gradio interface
interface = gr.Interface(
    fn=detect_objects,                     # Function to call for processing
    inputs=gr.Image(label="Upload Image", type="pil"),  # Input: Image in PIL format
    outputs=gr.Image(label="Detected Image"),           # Output: Annotated image
    title="YOLO Object Detection",                      # App title
    description="Upload an image to detect objects using YOLO."
)

# Launch the Gradio app
interface.launch()
