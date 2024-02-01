import argparse
import numpy as np
import pandas as pd
from skimage.measure import label, regionprops
from model import *
from dataset import *
from config import *


# Load and preprocess the input image
def preprocess_input(image_path: str, target_size: tuple = (256, 256)) -> np.ndarray:
    """
    Load and preprocess an input image.

    Args:
        image_path (str): Path to the input image.
        target_size (tuple): Target size for resizing. Default is (256, 256).

    Returns:
        np.ndarray: Preprocessed image.
    """
    image = cv2.imread(image_path)
    if image is None:
        print(f"Error: Unable to load image from {image_path}")
    if image.shape[0] == 0 or image.shape[1] == 0:
        print("Error: Loaded image has invalid dimensions.")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image / 255.0  # Normalize pixel values to the range [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image


def make_submission(folder_path: str, model: tf.keras.Model) -> pd.DataFrame:
    """
    Generate a submission DataFrame based on model predictions.

    Args:
        folder_path (str): Path to the folder containing test images.
        model (tf.keras.Model): Trained U-Net model.

    Returns:
        pd.DataFrame: DataFrame with ImageId and EncodedPixels columns for submission.
    """
    list_of_images = os.listdir(folder_path)
    image_id = []
    encoded_pixels = []

    for img_name in list_of_images:
        # Obtaining the model prediction.
        img = preprocess_input(os.path.join(folder_path, img_name))
        mask = model.predict(img, verbose=0)
        mask = np.squeeze(mask, axis=(0, 3))
        mask = cv2.resize(mask, (768, 768))
        mask = (mask > 0.3).astype(int)

        if np.all(mask == 0):
            image_id.append(img_name)
            encoded_pixels.append('')
        else:
            # Apply morphological operation to distinguish individual objects
            labeled_mask = label(mask)
            for region in regionprops(labeled_mask):
                # Create a mask for the current object
                single_ship_mask = (labeled_mask == region.label).astype(np.uint8)

                # Obtain RLE for the mask
                rle = rle_encode(single_ship_mask)

                # Add values to the lists
                image_id.append(img_name)
                encoded_pixels.append(rle)

    # Create a DataFrame
    df = pd.DataFrame({"ImageId": image_id, "EncodedPixels": encoded_pixels})
    return df


def main(args: dict) -> None:
    """
    Main function to initialize the U-Net model and generate a submission file.

    Args:
        args (dict): Dictionary containing command-line arguments or default values.
            Possible keys: 'saved_model_path'.
    """
    # Download model from google drive
    # Create and load model
    model = unet_model((IMAGE_HEIGHT, IMAGE_WIDTH, 3))
    model = tf.keras.models.load_model(os.path.join(SAVED_MODEL_PATH, 'model.keras'))
    # Create submission
    submission = make_submission(TEST_IMAGES, model)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    # Command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--saved_model_path", type=str, help='Specify folder path for saved model')
    parser.add_argument("--saved_submission_path", type=str, help='Specify folder path for saved submission.csv')
    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)
    # Calling the main function with parsed arguments
    main(args)