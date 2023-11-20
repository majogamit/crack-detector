from ultralytics import YOLO
import gradio as gr
from PIL import Image
import torch
import cv2
from utils.image_utils import preprocess_image, count_instance
from utils.data_utils import get_all_file_paths, generate_uuid
import numpy as np
import os

def load_model():
    """
    Load the YOLO model with pre-trained weights.
    
    Returns:
        model: Loaded YOLO model.
    """
    return YOLO('./weights/best.pt')
def detect_pattern(image_path):
    """
    Detect concrete cracks in the binary image.

    Parameters:
        image_path (str): Path to the binary image.

    Returns:
        tuple: Principal orientation and orientation category.
    """
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    skeleton = cv2.erode(image, np.ones((3, 3), dtype=np.uint8), iterations=1)
    contours, _ = cv2.findContours(skeleton, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    data_pts = np.vstack([contour.squeeze() for contour in contours])
    mean, eigenvectors = cv2.PCACompute(data_pts.astype(np.float32), mean=None)
    principal_orientation = np.arctan2(eigenvectors[0, 1], eigenvectors[0, 0])

    if -0.05 <= principal_orientation <= 0.05:
        orientation_category = "Horizontal"
    elif 1 <= principal_orientation <= 1.8:
        orientation_category = "Vertical"
    elif -0.99 <= principal_orientation <= 0.99:
        orientation_category = "Diagonal"
    else:
        orientation_category = "Other"

    return principal_orientation, orientation_category

def predict_segmentation(image, conf):
    """
    Perform segmentation prediction on a list of images.
    
    Parameters:
        image (list): List of images for segmentation.
        conf (float): Confidence score for prediction.

    Returns:
        tuple: Paths of the processed images, CSV file, DataFrame, and Markdown.
    """
    uuid = generate_uuid()
    image_list = [preprocess_image(Image.open(file.name)) for file in image]
    filenames = [file.name for file in image]
    conf= conf * 0.01
    model = load_model()
    results = model.predict(image_list, conf=conf, save=True, project='output', name=uuid, stream=True)
    processed_image_paths = []
    annotated_image_paths = []
    for i, r in enumerate(results):
        for m in r:
            masks = r.masks.data
            boxes = r.boxes.data
            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            people_masks = masks[people_indices]
            people_mask = torch.any(people_masks, dim=0).int() * 255
            processed_image_path = str(model.predictor.save_dir / f'binarize{i}.jpg')
            cv2.imwrite(processed_image_path, people_mask.cpu().numpy())
            processed_image_paths.append(processed_image_path)
            
            crack_image_path = processed_image_path
            principal_orientation, orientation_category = detect_pattern(crack_image_path)
            
            # Print the results if needed
            print(f"Crack Detection Results for {crack_image_path}:")
            print("Principal Component Analysis Orientation:", principal_orientation)
            print("Orientation Category:", orientation_category)

    csv, df = count_instance(results, filenames, uuid)

    csv = gr.File(value=csv, visible=True)
    df = gr.DataFrame(value=df, visible=True)
    md = gr.Markdown(visible=True)
    
    # # Delete binarized images after processing
    # for path in processed_image_paths:
    #     if os.path.exists(path):
    #         os.remove(path)
    
    return get_all_file_paths(f'output/{uuid}'), csv, df, md

