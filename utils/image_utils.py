import os
import numpy as np
import pandas as pd
from PIL import Image

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


def count_instance(result, filenames, uuid):
    """
    Counts the instances in the result and generates a CSV with the counts.
    
    Parameters:
        result (list): List containing results for each instance.
        filenames (list): Corresponding filenames for each result.
        uuid (str): Unique ID for the output folder name.
    
    Returns:
        tuple: Path to the generated CSV and dataframe with counts.
    """
    # Initializing the dataframe
    data = {
        'Index': [],
        'FileName': [],
        'Instance': []
    }
    df = pd.DataFrame(data)

    # Populate the dataframe with counts
    for i, res in enumerate(result):
        instance_count = len(res)
        df.loc[i] = [i, os.path.basename(filenames[i]), instance_count]

    # Save dataframe to a CSV file
    path = os.path.join('output', uuid)
    os.makedirs(path, exist_ok=True)
    csv_filename = os.path.join(path, '_results.csv')
    df.to_csv(csv_filename, index=False)

    return csv_filename, df
