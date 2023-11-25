from ultralytics import YOLO
import gradio as gr
from PIL import Image
import torch
import cv2
from utils.image_utils import preprocess_image, count_instance
from utils.data_utils import get_all_file_paths, generate_uuid

def load_model():
    """
    Load the YOLO model with pre-trained weights.
    
    Returns:
        model: Loaded YOLO model.
    """
    return YOLO('./weights/best.pt')

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
    for i, r in enumerate(results):
        for m in r:
        # print(r.masks)
        # get array results
            print(i)
            masks = r.masks.data
            boxes = r.boxes.data
            # extract classes
            clss = boxes[:, 5]
            # get indices of results where class is 0 (people in COCO)
            people_indices = torch.where(clss == 0)
            # use these indices to extract the relevant masks
            people_masks = masks[people_indices]
            # scale for visualizing results
            people_mask = torch.any(people_masks, dim=0).int() * 255
            # save to file
        cv2.imwrite(str(model.predictor.save_dir / f'merged_segs{i}.jpg'), people_mask.cpu().numpy()) 
    csv, df = count_instance(results, filenames, uuid)

    csv = gr.File(value=csv, visible=True)
    df = gr.DataFrame(value=df, visible=True)
    md = gr.Markdown(visible=True)

    return get_all_file_paths(f'output/{uuid}'), csv, df, md

