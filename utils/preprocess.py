import numpy as np
from PIL import Image

def preprocess_image(image):
    image = image.resize((224, 224))
    image = np.array(image) / 255.0
    image = image[np.newaxis, ...]
    return image
