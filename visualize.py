import matplotlib.pyplot as plt
import torch

def plot_training_history(history, num_epochs):
    """Plot training and validation loss and Dice score."""
    epochs_range = range(1, num_epochs + 1)
    plt.figure(figsize=(15, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, history['train_loss'], 'o-', label='Train Loss')
    plt.plot(epochs_range, history['val_loss'], 'o-', label='Validation Loss')
    plt.title('Loss Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, history['val_dsc'], 'o-', label='Validation DSC', color='orange')
    plt.title('Validation DSC Over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Dice Score')
    plt.xticks(epochs_range)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def visualize_sample_prediction(model, val_loader, device, num_classes):
    """Visualize a sample prediction from the last epoch."""
    model.eval()
    with torch.no_grad():
        sample_images, sample_masks = next(iter(val_loader))
        sample_images = sample_images.to(device)
        outputs = model(sample_images)
        predicted_masks = torch.argmax(outputs, dim=1)
    image_to_show = sample_images[0].cpu().permute(1, 2, 0).numpy()
    ground_truth_to_show = sample_masks[0].cpu().numpy()
    prediction_to_show = predicted_masks[0].cpu().numpy()
    if image_to_show.shape[2] == 3:
        image_to_show = image_to_show[:, :, 0]
    plt.figure(figsize=(18, 6))
    plt.subplot(1, 3, 1)
    plt.imshow(image_to_show, cmap='gray')
    plt.title("Input Image")
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(ground_truth_to_show, cmap='jet', vmin=0, vmax=num_classes-1)
    plt.title("Ground Truth Mask")
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(prediction_to_show, cmap='jet', vmin=0, vmax=num_classes-1)
    plt.title("Predicted Mask (Last Epoch)")
    plt.axis('off')
    plt.tight_layout()
    plt.show()