import argparse
import yaml
import mlflow
import mlflow.keras
from tensorflow.keras.models import load_model
from keras_preprocessing.image import ImageDataGenerator
from get_data import get_data
import os

def evaluate_model(config_path):
    # Load config
    config = get_data(config_path)

    # Extract config values
    model_path = config["model"]["sav_dir"]
    test_data_path = config["model"]["test_path"]
    img_size = tuple(config["model"]["image_size"])
    batch_size = config["img_augment"]["batch_size"]
    rescale = config["img_augment"]["rescale"]
    class_mode = config["img_augment"]["class_mode"]

    # Load model
    model = load_model(model_path)
    print(f"✅ Loaded model from {model_path}")

    # Set up test data generator
    test_datagen = ImageDataGenerator(rescale=rescale)
    test_generator = test_datagen.flow_from_directory(
        test_data_path,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=class_mode,
        shuffle=False
    )

    # Evaluate model
    loss, accuracy = model.evaluate(test_generator)
    print(f"📊 Evaluation results — Loss: {loss:.4f}, Accuracy: {accuracy:.4f}")

    # Log with MLflow
    mlflow_config = config["mlflow_config"]
    mlflow.set_tracking_uri(mlflow_config["remote_server_uri"])
    mlflow.set_experiment(mlflow_config["experiment_name"])

    with mlflow.start_run(run_name="model_evaluation"):
        mlflow.log_param("model_path", model_path)
        mlflow.log_metric("test_loss", loss)
        mlflow.log_metric("test_accuracy", accuracy)

        print("✅ Evaluation metrics logged to MLflow")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="params.yaml", help="Path to config file")
    args = parser.parse_args()
    evaluate_model(config_path=args.config)
