import argparse
import matplotlib.pyplot as plt
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from dataset import *
from config import *
from model import *


def train_loop(model: tf.keras.Model,
               train_generator: tf.keras.utils.Sequence,
               val_generator: tf.keras.utils.Sequence,
               epochs: int,
               loss: tf.keras.losses.Loss,
               optimizer: tf.keras.optimizers.Optimizer,
               saved_model_path: str) -> None:
    """
    Train the model using the provided generators and configurations.

    Parameters:
    - model (tf.keras.Model): The neural network model to be trained.
    - train_generator (tf.keras.utils.Sequence): The generator for training data.
    - val_generator (tf.keras.utils.Sequence): The generator for validation data.
    - epochs (int): Number of epochs for training.
    - loss (tf.keras.losses.Loss): The loss function to be used during training.
    - optimizer (tf.keras.optimizers.Optimizer): The optimizer for training.
    - saved_model_path (str): The path to save the trained model.

    Returns:
    - None
    """
    print(model.summary())
    reduceLROnPlate = ReduceLROnPlateau(monitor='val_loss', factor=0.33, patience=1, verbose=1, mode='min', min_delta=0.0001, cooldown=0, min_lr=1e-8)
    early_stopping = EarlyStopping(monitor="val_loss", mode="min", verbose=0, patience=10)

    callbacks_list = [reduceLROnPlate, early_stopping]
    model.compile(optimizer=optimizer, loss=loss, metrics=[dice_coefficient])
    steps_per_epoch = len(train_generator)

    history = model.fit(train_generator, validation_data=val_generator, epochs=epochs, steps_per_epoch=steps_per_epoch, callbacks=callbacks_list, shuffle=True, verbose=1)
    model.save(os.path.join(saved_model_path, 'model_v1.keras'))
    plt.figure(figsize=(16, 5))

    # Plot training and validation loss
    plt.subplot(1, 2, 1)
    plt.plot(range(epochs), history.history['loss'], 'bo-', label='Training loss')
    plt.plot(range(epochs), history.history['val_loss'], 'ro-', label='Validation loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Plot training and validation dice coefficient
    plt.subplot(1, 2, 2)
    plt.plot(range(epochs), history.history['dice_coefficient'], 'bo-', label='Training Dice Coefficient')
    plt.plot(range(epochs), history.history['val_dice_coefficient'], 'ro-', label='Validation Dice Coefficient')
    plt.title('Training and Validation Dice Coefficient')
    plt.xlabel('Epochs')
    plt.ylabel('Dice Coefficient')
    plt.legend()


def main(args: dict) -> None:
    """
    Main function to execute the training process.

    Parameters:
    - args (dict): Dictionary of command-line arguments.

    Returns:
    - None
    """
    epochs = args['epochs'] if args['epochs'] else EPOCHS
    lr = args['lr'] if args['lr'] else LR
    batch_size = args['batch_size'] if args['batch_size'] else BATCH_SIZE
    images_path = args['images_path'] if args['images_path'] else IMAGES_PATH
    masks_path = args['masks_path'] if args['masks_path'] else MASKS_PATH
    image_height = args["image_height"] if args["image_height"] else IMAGE_SIZE[0]
    image_width = args["image_width"] if args["image_width"] else IMAGE_SIZE[1]
    image_size = (image_height, image_width)
    saved_model_path = args['saved_model_path'] if args['saved_model_path'] else SAVED_MODEL_PATH

    optimizer = tf.optimizers.Adam(learning_rate=lr)
    loss = tf.losses.binary_crossentropy

    train_data, val_data = train_test_split_data(pd.read_csv(masks_path), empty_masks=2000, test_size=0.2)

    train_generator = AirBusDataset(image_folder=images_path,
                                          csv_file=train_data,
                                          batch_size=batch_size,
                                          image_size=image_size)

    val_generator = AirBusDataset(image_folder=images_path,
                                        csv_file=val_data,
                                         batch_size=batch_size,
                                         image_size=image_size)
    # Create model
    model = unet_model((image_height, image_width, 3))
    # Run train
    train_loop(model, train_generator, val_generator, epochs, loss, optimizer, saved_model_path)


if __name__ == "__main__":
    # Command-line argument parser
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_height", type=str, help='Specify height')
    parser.add_argument("--image_width", type=str, help='Specify width')
    parser.add_argument("--images_path", type=str, help='Specify path for train_v2 folder')
    parser.add_argument("--masks_path", type=str, help='Specify path for train_ship_segmentations_v2.csv file')
    parser.add_argument("--epochs", type=int, help='Specify epoch for model training')
    parser.add_argument("--lr", type=float, help='Specify learning rate')
    parser.add_argument("--batch_size", type=float, help='Specify batch size for training')
    parser.add_argument("--saved_model_path", type=str, help='Specify folder path for saved model')

    # Parsing command-line arguments
    args = parser.parse_args()
    args = vars(args)

    # Calling the main function with parsed arguments
    main(args)