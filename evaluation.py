from ultralytics import YOLO

def evaluate_model(weights_path, data_yaml_path):
    """
    Load a YOLOv8 model from weights and run validation to inspect metrics.
    """
    print(f"Loading model weights from: {weights_path}")
    model = YOLO(weights_path)
    
    print(f"Running validation using dataset config: {data_yaml_path}")
    # Run validation on the validation/test split
    metrics = model.val(data=data_yaml_path, imgsz=640, batch=16)
    
    # Print out key performance metrics
    print("\n--- Model Evaluation Results ---")
    print(f"mAP50-95 : {metrics.box.map:.4f}")
    print(f"mAP50    : {metrics.box.map50:.4f}     ")
    print(f"Precision: {metrics.box.mean_results()[0]:.4f}")
    print(f"Recall   : {metrics.box.mean_results()[1]:.4f}")
    
    return metrics

if __name__ == "__main__":
    # Paths based on your project configuration
    WEIGHTS_FILE = "best.pt"  # Change to "last.pt" if you want to test the last checkpoint
    DATA_YAML = r"C:\Users\VANSH\Projects\solar-sweeper-ai\data\solar_final_resplit\data.yaml"
    
    evaluate_model(WEIGHTS_FILE, DATA_YAML)