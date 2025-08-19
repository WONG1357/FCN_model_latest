# FCN-PyTorch Segmentation Project

This repository contains a PyTorch implementation of a Fully Convolutional Network (FCN) for image segmentation, designed to work with `.npy` datasets stored on Google Drive. The project is modularized into separate scripts for dataset handling, model setup, training, evaluation, and visualization, making it easy to understand and extend.

## Project Structure

```
FCN-pytorch-project/
├── dataset.py           # Custom dataset class for loading .npy files
├── model_utils.py       # Model initialization and setup
├── train.py             # Training and validation loop
├── evaluate.py          # Prediction generation and evaluation
├── visualize.py         # Visualization of training history and predictions
├── main.py              # Main script to run the pipeline
├── requirements.txt     # Python dependencies
├── README.md           # This file
```

## Setup Instructions

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/your-username/FCN-pytorch-project.git
   cd FCN-pytorch-project
   ```

2. **Install Dependencies**:
   Ensure you have Python 3.8+ installed. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. **Prepare Data**:
   - Place your `.npy` files (`X_train.npy`, `y_train.npy`, `X_val.npy`, `y_val.npy`, `X_test.npy`, `y_test.npy`) in a directory accessible via Google Drive (e.g., `/content/drive/MyDrive/intern RF transverse latest file/`).
   - Update the `base_path` variable in `main.py` if your data is stored in a different location.

4. **Run the Pipeline**:
   Execute the main script to train the model, evaluate it, and generate visualizations:
   ```bash
   python main.py
   ```

## Requirements

- Python 3.8+
- PyTorch 2.0.0 or higher
- NumPy
- Matplotlib
- SciPy
- Google Colab (for Google Drive integration)

See `requirements.txt` for detailed versions.

## Usage

The `main.py` script orchestrates the entire pipeline:
- Loads and preprocesses `.npy` datasets using `dataset.py`.
- Initializes the FCN8s model and VGGNet backbone using `model_utils.py`.
- Trains and validates the model using `train.py`.
- Generates and saves predictions as JPG images using `evaluate.py`.
- Visualizes training history and sample predictions using `visualize.py`.

Outputs, including model checkpoints and prediction images, are saved to `/content/drive/MyDrive/internship models/FCN/`.

## Notes

- The project clones the `pochih/FCN-pytorch` repository for the FCN8s and VGGNet implementations.
- The model uses a GPU if available; otherwise, it falls back to CPU.
- The dataset is assumed to consist of grayscale images and segmentation masks stored as `.npy` files.
- The code includes post-processing of segmentation masks to improve prediction quality.
- Dice scores are computed to evaluate model performance before and after post-processing.

## License

This project is licensed under the MIT License.