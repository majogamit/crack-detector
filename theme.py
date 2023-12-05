import os
import shutil
import numpy as np
from PIL import Image
import cv2
import torch
from ultralytics import YOLO
from torchvision.transforms.functional import resize

def preprocess_image(image_path):
    """ 
    Preprocesses the input image.
    
    Parameters:
        image_path (str): Path to the image to preprocess.
    
    Returns:
        numpy.array: Resized and converted RGB version of the input image.
    """
    # Open the image from the provided path
    image = Image.open(image_path)
    np_image = np.array(image)
    return np_image

def resize_mask(mask, target_size):
    """ 
    Resize the mask to the target size.
    
    Parameters:
        mask (numpy.array): Binarized mask.
        target_size (tuple): Target size (height, width).
    
    Returns:
        numpy.array: Resized mask.
    """
    pil_mask = Image.fromarray(mask)
    resized_mask = resize(pil_mask, target_size, interpolation=Image.NEAREST)
    return np.array(resized_mask)

images = ['diagonal2.png', 'diagonal3.webp', 'vertical3 - Copy.jpg']
images_np = [preprocess_image(image) for image in images]

model = YOLO('./weights/best.pt')

for i, image_path in enumerate(images):
    results = model.predict(image_path, save=True, project='output', name=f'wow{i}', stream=True)
    for r in results:
        instance_count = len(r)
        print(r.orig_shape)
        if r.masks is not None and r.masks.data.numel() > 0:
            masks = r.masks.data
            boxes = r.boxes.data
            clss = boxes[:, 5]
            people_indices = torch.where(clss == 0)
            people_masks = masks[people_indices]
            people_mask = torch.any(people_masks,dim=0).int() * 255

            # Resize the binarized mask to match the original image size
            # resized_mask = resize_mask(people_mask.cpu().numpy(), images_np[i].shape[:2])
            resized_mask = people_mask.numpy()
            processed_image_path = str(f'output/wow0/binarize{i}.jpg')
            cv2.imwrite(processed_image_path, resized_mask)
    
    the_paths = f'output/wow{i}/{os.path.basename(image_path)}'
    print(the_paths)
    shutil.copyfile(the_paths, f'output/wow0/image{i}.jpg')