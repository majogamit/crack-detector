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
from IPython.core.display import display,HTML

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
                
                # Display section for results
                # md_result = gr.Markdown("**Results**", visible=False)
                # csv_image = gr.File(label='CSV File', interactive=False, visible=False)
                # df_image = gr.DataFrame(visible=False)
        image_reference = gr.Image(sources=['upload', 'webcam'],
                                       label='Reference Image',)
    # Video tab
    with gr.Tab("Video"):
        with gr.Row():
            with gr.Column():
                # Input section for uploading images
                video_input = gr.Video(
                    label="Video Input",
                    format='mp4'
                )
                gr.Examples(["examples/IMG_3636.mp4"],
                            inputs=video_input)
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
                video_results = gr.Textbox(label="Result")
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
    
    
    
    def count_instance(result, filenames, uuid, width_list, orientation_list, image_path):
        """
        Counts the instances in the result and generates a CSV with the counts.

        Parameters:
            result (list): List containing results for each instance.
            filenames (list): Corresponding filenames for each result.
            uuid (str): Unique ID for the output folder name.
            width_list (list): List containing width values for each instance.
            orientation_list (list): List containing orientation values for each instance.

        Returns:
            tuple: Path to the generated CSV and dataframe with counts.
        """
        # Initializing the dataframe
        data = {
            'Index': [],
            'FileName': [],
            'Orientation': [],
            'Width': [],
            'Instance': []
        }
        df = pd.DataFrame(data)

        # Populate the dataframe with counts, width, and orientation
        for i, res in enumerate(result):
            instance_count = len(res)
            df.loc[i] = [i, os.path.basename(filenames[i]), orientation_list[i], width_list[i], instance_count]

        # Save dataframe to a CSV file
        path = os.path.join('output', uuid)
        os.makedirs(path, exist_ok=True)
        csv_filename = os.path.join(path, '_results.csv')

        # Reorder columns
        df = df[['Index', 'FileName', 'Orientation', 'Width', 'Instance']]

        # Create a new dataframe (df2) with all columns from df
        df2 = df.copy()

        # Add another column for the image (modify as per your requirement)
        image_column = [image_path[i].format(i) for i in range(len(df))]
        df2['Image'] = image_column

        # Save the modified dataframe to a CSV file
        # df2.to_csv(csv_filename, index=False)
        html_table = HTML(df.to_html(escape=False))
        display(html_table)
        return csv_filename, df2

# Example usage:
# count_instance(result, filenames, uuid, width_list, orientation_list)



    def predict_segmentation_im(image, conf, reference):
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
        output_image_paths = []
        result_list = []
        width_list = []
        orientation_list = []
        # Populate the dataframe with counts
        for i, r in enumerate(results):
            result_list.append(r)
            instance_count = len(r)
            # for m in r:
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

            # Delete binarized and initial segmented images after processing
            for path in processed_image_paths:
                if os.path.exists(path):
                    os.remove(path)
            
            res = f"Pattern: {orientation_category}\nWidth: {width}\nLength:\nCrack Instance: {instance_count}\nSafety Recommendation:"
        # results = gr.Textbox(res, visible=True)
        csv, df = count_instance(result_list, filenames, uuid, width_list, orientation_list, output_image_paths)
        print(len(result_list))
        print(filenames)
        print("DF")
        print(df)
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
            gr.Slider(value=20)
        ],
        outputs=[image_input, image_output, md_result, csv_image, df_image, conf]
    )

# Launch the Gradio app
demo.launch()