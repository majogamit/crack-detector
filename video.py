import cv2
import gradio as gr
import shortuuid
from ultralytics import YOLO
from utils.data_utils import clear_all
import torch
import numpy as np
import os

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

    # Video tab
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                # Input section for uploading images
                video_input = gr.Video(
                    label="Video Input",
                    format='mp4'
                )

                #Confidence Score for prediction
                conf = gr.Slider(value=20,step=5, label="Confidence",
                                   interactive=True)
                # distance = gr.Slider(value=5,step=5, label="Distance (m)",
                #                    interactive=True)
                # Buttons for segmentation and clearing
                with gr.Row():
                    video_button = gr.Button("Segment", variant='primary')
                    video_clear = gr.ClearButton()

            with gr.Column():
                # Display section for segmented videos
                video_output = gr.Video(label="Video Output")
                # video_results = gr.Textbox(label="Result")
                # Display section for results
                # md_result = gr.Markdown("**Results**", visible=False)
                # csv_video = gr.File(label='CSV File', interactive=False, visible=False)
                # df_video = gr.DataFrame(visible=False)
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


    def predict_segmentation_vid(video, conf):
            """
            Perform segmentation prediction on a list of images.
            
            Parameters:
                image (list): List of images for segmentation.
                conf (float): Confidence score for prediction.

            Returns:
                tuple: Paths of the processed images, CSV file, DataFrame, and Markdown.
            """
            uuid = str(shortuuid.uuid())
            conf= conf * 0.01
            # filename = image.name
            results = model.predict(video, conf=conf, save=True, project='output', name=uuid, stream=True)
            processed_image_paths = []
            annotated_image_paths = []
            # Populate the dataframe with 
            for result in results:
                boxes = result.boxes  # Boxes object for bbox outputs
                masks = result.masks  # Masks object for segmentation masks outputs
                probs = result.probs  # Probs object for classification outputs
            # for i, r in enumerate(results):
            #     instance_count = len(r)
            #     for m in r:
            #         masks = r.masks.data
            #         boxes = r.boxes.data
            #         clss = boxes[:, 5]
            #         people_indices = torch.where(clss == 0)
            #         people_masks = masks[people_indices]
            #         people_mask = torch.any(people_masks, dim=0).int() * 255
            #         processed_image_path = str(model.predictor.save_dir / f'binarize{i}.jpg')
            #         cv2.imwrite(processed_image_path, people_mask.cpu().numpy())
            #         processed_image_paths.append(processed_image_path)
                    
            #         crack_image_path = processed_image_path
            #         principal_orientation, orientation_category = detect_pattern(crack_image_path)
                    
            #         # Print the results if needed
            #         print(f"Crack Detection Results for {crack_image_path}:")
            #         print("Principal Component Analysis Orientation:", principal_orientation)
            #         print("Orientation Category:", orientation_category)
            # csv = gr.File(value=csv, visible=True)
            # df = gr.DataFrame(value=df, visible=True)
            # md = gr.Markdown(visible=True)
            
            # # Delete binarized images after processing
            # for path in processed_image_paths:
            #     if os.path.exists(path):
            #         os.remove(path)
            
            # res = f"Pattern: {orientation_category}\nWidth:\nLength:\nCrack Instance: {instance_count}\nSafety Recommendation:"
            # results = gr.Textbox(res, visible=True)
            return get_all_file_paths(f'output/{uuid}')


        # Connect the buttons to the prediction function and clear function
    video_button.click(
        predict_segmentation_vid,
        inputs=[video_input, conf],
        outputs=[video_output],
    )
    # video_button.click(
    #     sample,
    #     inputs=[video_input],
    #     outputs=[video_output]
    # )
    video_clear.click(
        lambda: [
            None,
            None,
            gr.Slider(value=20),
            None
        ],
        outputs=[video_input, video_output, conf]
    )

    # Launch the Gradio app
    demo.launch()