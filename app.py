import gradio as gr
from utils.data_utils import clear_all
from utils.model_utils import predict_segmentation

# Clear any previous data and configurations
clear_all()

# Define the color scheme/theme for the website
theme = gr.themes.Soft(
    primary_hue=gr.themes.Color(c100="#dbeafe", 
                                c200="#bfdbfe", 
                                c300="#93c5fd", 
                                c400="#60a5fa", 
                                c50="#eff6ff", 
                                c500="#3b82f6", 
                                c600="#2563eb", 
                                c700="#1d4ed8", 
                                c800="#1e40af", 
                                c900="#00037c", 
                                c950="#1d3660"),
    secondary_hue=gr.themes.Color(c100="#fff6cc", 
                                  c200="#fff2b2", 
                                  c300="#ffee99", 
                                  c400="#ffe97f", 
                                  c50="#fffae5", 
                                  c500="#ffe566", 
                                  c600="#ffe14c", 
                                  c700="#ffdd32", 
                                  c800="#ffd819", 
                                  c900="#ffd400", 
                                  c950="#ffbf00"),
    neutral_hue="neutral",
    font=[gr.themes.GoogleFont('Inter'), 'ui-sans-serif', 'system-ui', 'sans-serif'],
).set(
    block_info_text_color="*neutral_950",
    block_label_text_color="*primary_900",
    block_title_text_color="*primary_900",
    border_color_primary="*neutral_300",
    button_primary_background_fill="*primary_900",
    button_primary_background_fill_hover="*primary_700",
    button_secondary_background_fill="*secondary_950",
    button_secondary_background_fill_hover="*secondary_600",
    block_background_fill="*neutral_100",
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
    gr.Markdown("# Corrosion Segmentation")
    gr.Markdown("Upload corrosion images and get segmented corrosion results.")

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