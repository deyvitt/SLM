from PIL import Image
import numpy as np

def preprocess_image(image_path):
    # Open image
    image = Image.open(image_path)
    # Resize
    image = image.resize((224, 224))
    # Convert to numpy array
    image = np.array(image)
    # Normalize
    image = image / 255.0
    return image