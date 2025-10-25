# test_model.py
import tensorflow as tf

def evaluate_model(model_path="model/my_model.keras", dataset_dir="curacha_dataset"):
    """
    Loads the model and evaluates its accuracy on the validation/test split.
    Assumes dataset_dir has subfolders per class.
    """
    # Load model
    model = tf.keras.models.load_model(model_path)

    # Load test data (20% split)
    test_ds = tf.keras.utils.image_dataset_from_directory(
        dataset_dir,
        image_size=(180, 180),
        batch_size=32,
        shuffle=True,
        validation_split=0.2,
        subset="validation",
        seed=123
    )

    # Evaluate
    loss, accuracy = model.evaluate(test_ds)
    return {"loss": float(loss), "accuracy": float(accuracy)}
