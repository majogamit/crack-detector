from ultralytics import YOLO
import gradio as gr
from PIL import Image
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
    results = model.predict(image_list, conf=conf, save=True, project='output', name=uuid)

    csv, df = count_instance(results, filenames, uuid)

    csv = gr.File(value=csv, visible=True)
    df = gr.DataFrame(value=df, visible=True)
    md = gr.Markdown(visible=True)

    return get_all_file_paths(f'output/{uuid}'), csv, df, md

