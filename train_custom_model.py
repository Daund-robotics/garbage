from ultralytics import YOLO
import os

# Configuration
DATA_YAML = r"D:\garbage updated\dataset\data.yaml"
MODEL_NAME = "yolov8n.pt"  # Load a pretrained model (recommended for training)
EPOCHS = 50
IMG_SIZE = 640

def train():
    # Load a model
    model = YOLO(MODEL_NAME)  # load a pretrained model (recommended for training)

    # Train the model
    # workers=0 is often safer on Windows to avoid multiprocessing issues
    results = model.train(data=DATA_YAML, epochs=EPOCHS, imgsz=IMG_SIZE, workers=0) 
    
    print("Training Complete!")
    print(f"Best model saved at: {results.save_dir}")

if __name__ == "__main__":
    # Ensure ultralytics is installed
    try:
        import ultralytics
        print(f"Ultralytics version: {ultralytics.__version__}")
        train()
    except ImportError:
        print("Please install ultralytics first: pip install ultralytics")
