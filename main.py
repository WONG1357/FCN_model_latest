import os
import torch
from torch.utils.data import DataLoader
import numpy as np
from google.colab import drive
from dataset import CustomNumpyDataset
from model_utils import setup_fcn_model, setup_optimizer, setup_criterion
from train import train_model
from evaluate import save_predictions_as_jpg
from visualize import plot_training_history, visualize_sample_prediction

# Mount Google Drive
print("--- Mounting Google Drive ---")
try:
    drive.mount('/content/drive', force_remount=True)
    print("✅ Google Drive mounted successfully.")
except Exception as e:
    print(f"❌ Error mounting drive: {e}")

# Define dataset file paths
print("\n--- Defining dataset file paths ---")
base_path = '/content/drive/MyDrive/intern RF transverse latest file/'
X_train_path = os.path.join(base_path, 'X_train.npy')
y_train_path = os.path.join(base_path, 'y_train.npy')
X_val_path = os.path.join(base_path, 'X_val.npy')
y_val_path = os.path.join(base_path, 'y_val.npy')
X_test_path = os.path.join(base_path, 'X_test.npy')
y_test_path = os.path.join(base_path, 'y_test.npy')
print("✅ File paths defined for train, validation, and test sets.")

# Configuration
temp_mask = np.load(y_train_path)
NUM_CLASSES = int(np.max(temp_mask)) + 1
del temp_mask
BATCH_SIZE = 8
NUM_EPOCHS = 10
LEARNING_RATE = 1e-4
device = "cuda" if torch.cuda.is_available() else "cpu"
MODEL_SAVE_DIR = '/content/drive/MyDrive/internship models/FCN'
MODEL_SAVE_PATH = os.path.join(MODEL_SAVE_DIR, 'best_fcn_model.pth')
os.makedirs(MODEL_SAVE_DIR, exist_ok=True)

print("\n--- Configuration ---")
print(f"Device: {device}")
print(f"Number of Classes: {NUM_CLASSES}")
print(f"Batch Size: {BATCH_SIZE}")
print(f"Epochs: {NUM_EPOCHS}")
print(f"Model Save Path: {MODEL_SAVE_PATH}")

# Create datasets and dataloaders
print("\n--- Creating datasets and dataloaders ---")
train_dataset = CustomNumpyDataset(images_path=X_train_path, masks_path=y_train_path)
val_dataset = CustomNumpyDataset(images_path=X_val_path, masks_path=y_val_path)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2, pin_memory=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
print("✅ Dataloaders are ready.")

# Create test dataset and dataloader
print("\n--- Creating test dataset and dataloader ---")
try:
    test_dataset = CustomNumpyDataset(images_path=X_test_path, masks_path=y_test_path)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
    print("✅ Test dataloader is ready.")
except FileNotFoundError:
    print(f"❌ WARNING: Test files not found at {X_test_path} or {y_test_path}. Skipping test loader creation.")
    test_loader = None

# Initialize model, loss, and optimizer
print("\n--- Initializing Model ---")
model = setup_fcn_model(NUM_CLASSES, device)
criterion = setup_criterion()
optimizer = setup_optimizer(model, LEARNING_RATE)
print("✅ Model, Loss, and Optimizer are ready.")

# Train the model
history = train_model(model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, device, MODEL_SAVE_PATH, NUM_CLASSES)

# Plot training history
print("\n--- Plotting Training History ---")
plot_training_history(history, NUM_EPOCHS)

# Visualize a sample prediction
print("\n--- Visualizing a sample prediction ---")
visualize_sample_prediction(model, val_loader, device, NUM_CLASSES)

# Load best model
print("\n--- Loading best performing model ---")
best_model = setup_fcn_model(NUM_CLASSES, device)
best_model.load_state_dict(torch.load(MODEL_SAVE_PATH))
best_model.eval()
print(f"✅ Best model loaded from {MODEL_SAVE_PATH}")

# Generate and save predictions
print("\n--- Generating and saving predictions ---")
internal_val_dir = '/content/drive/MyDrive/internship models/FCN/internal validation (on train set)'
external_val_dir = '/content/drive/MyDrive/internship models/FCN/external validation (on test set)'
os.makedirs(internal_val_dir, exist_ok=True)
os.makedirs(external_val_dir, exist_ok=True)
print(f"✅ Output directories are ready: {internal_val_dir}, {external_val_dir}")

print("\n--- Generating comparison images for training set ---")
train_loader_no_shuffle = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2, pin_memory=True)
dice_before_train, dice_after_train = save_predictions_as_jpg(best_model, train_loader_no_shuffle, device, internal_val_dir, NUM_CLASSES)
print(f"\n--- Train Set Evaluation Complete ---")
print(f"Total Train Images Processed: {len(train_dataset)}")
print(f"Average Dice (Before Post-Processing): {dice_before_train:.4f}")
print(f"Average Dice (After Post-Processing):  {dice_after_train:.4f}")

if test_loader:
    print("\n--- Generating comparison images for test set ---")
    dice_before_test, dice_after_test = save_predictions_as_jpg(best_model, test_loader, device, external_val_dir, NUM_CLASSES)
    print(f"\n--- Test Set Evaluation Complete ---")
    print(f"Total Test Images Processed: {len(test_dataset)}")
    print(f"Average Dice (Before Post-Processing): {dice_before_test:.4f}")
    print(f"Average Dice (After Post-Processing):  {dice_after_test:.4f}")
else:
    print("SKIPPED: Test loader was not created.")

print("\n--- All comparison images have been generated and saved successfully! ---")
