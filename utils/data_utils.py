import os
import shutil
import shortuuid


def clear_all():
    """
    Removes the 'output/' directory along with its content.
    
    Returns:
        None
    """
    shutil.rmtree('output/', ignore_errors=True)

def clear_value_tab(path):
    """
    Removes a specific sub-directory under 'output/'.
    
    Parameters:
        path (str): Sub-directory to remove.
    
    Returns:
        None
    """
    print(path)
    shutil.rmtree(os.path.join('output/', path), ignore_errors=True)

def get_all_file_paths(directory):
    """
    Collects all image file paths under a given directory.
    
    Parameters:
        directory (str): Directory to search for image files.
    
    Returns: 
        list: List of image file paths.
    """
    allowed_extensions = ('.png', '.jpg', '.jpeg', '.gif', '.bmp')
    return [
        os.path.join(root, file)
        for root, _, files in os.walk(directory)
        for file in files
        if file.lower().endswith(allowed_extensions)
    ]

def generate_uuid():
    """
    Generates a short unique identifier.
    
    Returns:
        str: Unique identifier string.
    """
    return str(shortuuid.uuid())
