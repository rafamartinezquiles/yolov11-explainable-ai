import argparse
from ultralytics import YOLO

def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="YOLOv11 Batch Prediction")
    parser.add_argument('--model', type=str, required=True, help="Path to the YOLOv11 model file.")
    parser.add_argument('--source', type=str, required=True, help="Path to the source folder containing images for prediction.")
    
    args = parser.parse_args()

    # Load the YOLOv11 model
    print(f"Loading YOLOv11 model from: {args.model}")
    model = YOLO(args.model)

    # Perform predictions on the source images
    print(f"Starting predictions on images in: {args.source}")
    model.predict(source=args.source, save_txt=True)
    print("Predictions completed. Results saved.")

if __name__ == "__main__":
    main()
