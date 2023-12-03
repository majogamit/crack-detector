import cv2
import gradio as gr
import pandas as pd
import shortuuid
from ultralytics import YOLO
from utils.data_utils import clear_all
import torch
import numpy as np
import os
from utils.measure_utils import ContourAnalyzer
from PIL import Image
from IPython.display import display,HTML
import plot as pt
# Clear any previous data and configurations
clear_all()
model = YOLO('./weights/best.pt')
# Define the color scheme/theme for the website
theme = gr.themes.Soft(
    primary_hue="orange",
    secondary_hue="sky",
)
#Custom css for styling
css = """
    .size {
        min-height: 400px !important;
        max-height: 400px !important;
        overflow: auto !important;
    }
"""

# Create the Gradio interface using defined theme and CSS
with gr.Blocks(theme=theme, css=css) as demo:
    # Title and description for the app
    gr.Markdown("# Concrete Crack Detection and Segmentation")
    gr.Markdown("Upload concrete crack images and get segmented results.")

    # Image tab
    with gr.Tab("Image"):
        
        with gr.Row():           
            with gr.Column():
                # Input section for uploading images
                image_input = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Image Input",
                    elem_classes="size",
                )


                #Confidence Score for prediction
                conf = gr.Slider(value=20,step=5, label="Confidence", 
                                   interactive=True)
                distance = gr.Slider(value=5,step=5, label="Distance (m)", 
                                   interactive=True)
                # Buttons for segmentation and clearing
                with gr.Row():
                    image_button = gr.Button("Segment", variant='primary')
                    image_clear = gr.ClearButton()

            with gr.Column():
                # Display section for segmented images
                image_output = gr.Gallery(
                    label="Image Output",
                    show_label=True,
                    elem_id="gallery",
                    columns=2,
                    object_fit="contain",
                    height=400,
                )
                md_result = gr.Markdown("**Results**", visible=False)
                csv_image = gr.File(label='CSV File', interactive=False, visible=False)
                df_image = gr.DataFrame(visible=False)

                image_results = gr.Textbox(label="Result")
                
        image_reference = gr.File(
                    file_count="multiple",
                    file_types=["image"],
                    label="Reference Image",)


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

    def load_model():
        """
        Load the YOLO model with pre-trained weights.
        
        Returns:
            model: Loaded YOLO model.
        """
        return YOLO('./weights/best.pt')
    
    def generate_uuid():
        """
        Generates a short unique identifier.
        
        Returns:
            str: Unique identifier string.
        """
        return str(shortuuid.uuid())


    def preprocess_image(image):
        """ 
        Preprocesses the input image.
        
        Parameters:
            image (numpy.array or PIL.Image): Image to preprocess.
        
        Returns:
            numpy.array: Resized and converted RGB version of the input image.
        """
        # Convert PIL image to numpy array if required
        if isinstance(image, Image.Image):
            image = np.array(image)

        # Resize and convert the image to RGB
        input_image = Image.fromarray(image)
        input_image = input_image.resize((640, 640))
        input_image = input_image.convert("RGB")

        return np.array(input_image)
    
    
    def predict_segmentation_im(image, conf, reference):
        """
        Perform segmentation prediction on a list of images.
        
        Parameters:
            image (list): List of images for segmentation.
            conf (float): Confidence score for prediction.

        Returns:
            tuple: Paths of the processed images, CSV file, DataFrame, and Markdown.
        """
        print("THE REFERENCE IN APPPY", reference)
        uuid = generate_uuid()
        image_list = [preprocess_image(Image.open(file.name)) for file in image]
        filenames = [file.name for file in image]
        conf= conf * 0.01
        model = load_model()
        results = model.predict(image_list, conf=conf, save=True, project='output', name=uuid, stream=True)
        processed_image_paths = []
        output_image_paths = []
        result_list = []
        width_list = []
        orientation_list = []
        # Populate the dataframe with counts
        for i, r in enumerate(results):
            result_list.append(r)
            instance_count = len(r)
            if r.masks is not None and r.masks.data.numel() > 0:
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

                # Load the original image in color
                original_img = cv2.imread(f'output/{uuid}/image{i}.jpg')
                orig_image_path = str(model.predictor.save_dir / f'image{i}.jpg')
                processed_image_paths.append(orig_image_path)
                # Load and resize the binary image to match the dimensions of the original image
                binary_image = cv2.imread(f'output/{uuid}/binarize{i}.jpg', cv2.IMREAD_GRAYSCALE)
                binary_image = cv2.resize(binary_image, (original_img.shape[1], original_img.shape[0]))

                contour_analyzer = ContourAnalyzer()
                max_width, thickest_section, thickest_points, distance_transforms = contour_analyzer.find_contours(binary_image)

                visualized_image = original_img.copy()
                cv2.drawContours(visualized_image, [thickest_section], 0, (0, 255, 0), 1)

                contour_analyzer.draw_circle_on_image(visualized_image, (int(thickest_points[0]), int(thickest_points[1])), 5, (57, 255, 20), -1)
                print("Max Width in pixels: ", max_width)

                width = contour_analyzer.calculate_width(y=10, x=5, pixel_width=max_width, calibration_factor=0.001, distance=150)
                print("Max Width, converted: ", width)
                
                visualized_image_path = f'output/{uuid}/visualized_image{i}.jpg'
                output_image_paths.append(visualized_image_path)
                cv2.imwrite(visualized_image_path, visualized_image)

                width_list.append(round(width, 2))
                orientation_list.append(orientation_category)
            else:
                original_img = cv2.imread(f'output/{uuid}/image{i}.jpg')
                visualized_image_path = f'output/{uuid}/visualized_image{i}.jpg'
                output_image_paths.append(visualized_image_path)
                cv2.imwrite(visualized_image_path, original_img)
                width_list.append('None')
                orientation_list.append('None')

        # Delete binarized and initial segmented images after processing
        for path in processed_image_paths:
            if os.path.exists(path):
                os.remove(path)
                
        # results = gr.Textbox(res, visible=True)
        csv, df = pt.count_instance(result_list, filenames, uuid, width_list, orientation_list, output_image_paths, reference)

        csv = gr.File(value=csv, visible=True)
        df = gr.DataFrame(value=df, visible=True)
        md = gr.Markdown(visible=True)

        # return get_all_file_paths(f"output/{uuid}/"), csv, df, md
        return output_image_paths, csv, df, md

    def get_all_file_paths(directory):
        """
        Collects all image file paths under a given directory.
        
        Parameters:
            directory (str): Directory to search for image files.
        
        Returns: 
            str: String of image file paths separated by a newline character.
        """
        allowed_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.mp4', '.avi', '.mov')
        
        file_paths = [
            os.path.join(root, file)
            for root, _, files in os.walk(directory)
            for file in files
            if file.lower().endswith(allowed_extensions)
        ]
        
        return '\n'.join(file_paths)

    # Connect the buttons to the prediction function and clear function
    image_button.click(
        predict_segmentation_im,
        inputs=[image_input, conf, image_reference],
        outputs=[image_output, csv_image, df_image, md_result]
    )
    
    image_clear.click(
        lambda: [
            None,
            None,
            gr.Markdown(visible=False),
            gr.File(visible=False),
            gr.DataFrame(visible=False),
            gr.Slider(value=20),
            None
        ],
        outputs=[image_input, image_output, md_result, csv_image, df_image, conf, image_reference]
    )

# Launch the Gradio app
demo.launch()