import gradio as gr
from utils.data_utils import clear_all
from utils.model_utils import predict_segmentation

# Clear any previous data and configurations
clear_all()

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
                
                # Display section for results
                md_result = gr.Markdown("**Results**", visible=False)
                csv_image = gr.File(label='CSV File', interactive=False, visible=False)
                df_image = gr.DataFrame(visible=False)
            

    # Connect the buttons to the prediction function and clear function
    image_button.click(
        predict_segmentation,
        inputs=[image_input, conf],
        outputs=[image_output, csv_image, df_image, md_result]
    )
    
    image_clear.click(
        lambda: [
            None,
            None,
            gr.Markdown.update(visible=False),
            gr.File.update(visible=False),
            gr.DataFrame.update(visible=False),
            gr.Slider.update(value=20)
        ],
        outputs=[image_input, image_output, md_result, csv_image, df_image, conf]
    )

# Launch the Gradio app
demo.launch()