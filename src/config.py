import tensorflow as tf
import numpy as np
import gdown
import os
from tensorflow.keras import backend as K
from typing import Union


def load_model(url: str, output_path: str):
    """
        Download a model from the given URL and save it to the specified output path
        if the model does not exist at the output path.

        Args:
            url (str): The URL from which to download the model.
            output_path (str): The path to save the downloaded model.
        Return: None
        """
    if not os.path.exists(output_path):
        gdown.download(url, output_path, quiet=False)
        print("Model loaded!")
    else:
        print(f"Model '{output_path}' already exists.")


@tf.keras.utils.register_keras_serializable(package='Custom', name='dice_coefficient')
def dice_coefficient(y_true: Union[tf.Tensor, np.ndarray],
                     y_pred: Union[tf.Tensor, np.ndarray],
                     smooth: float = 1e-5) -> Union[tf.Tensor, np.ndarray]:
    """
    Compute the Dice Coefficient metric.

    Parameters:
    - y_true (Union[tf.Tensor, np.ndarray]): Ground truth binary mask.
    - y_pred (Union[tf.Tensor, np.ndarray]): Predicted binary mask.
    - smooth (float, optional): Smoothing factor to avoid division by zero. Default is 1e-5.

    Returns:
    - Union[tf.Tensor, np.ndarray]: Dice Coefficient score.
    """
    intersection = K.sum(K.abs(y_true * y_pred), axis=-1)
    return (2. * intersection + smooth) / (K.sum(K.square(y_true), -1) + K.sum(K.square(y_pred), -1) + smooth)

# Global variables
BATCH_SIZE = 16
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
IMAGES_PATH = './data/train_v2'
TEST_IMAGES = './data/train_v2'
MASKS_PATH = './data/train_ship_segmentations_v2.csv'
EPOCHS = 20
IMAGE_SIZE = (IMAGE_HEIGHT, IMAGE_WIDTH)
LR = 0.001
SAVED_MODEL_PATH = "."
SAVED_SUBMISSION_PATH = "."
MODEL_URL = 'https://drive.google.com/uc?export=download&id=1YXSgKxWiUmhwdf8PemtDd1UurNuobJ1W'
