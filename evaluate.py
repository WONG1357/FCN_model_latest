import torch
import os
from scipy import ndimage
import matplotlib.pyplot as plt

def post_process_mask(mask):
    """Applies post-processing to a binary segmentation mask."""
    labels, num_features = ndimage.label(mask)
    if num_features == 0:
        return mask
    component_sizes = np.bincount(labels.ravel())
    if len(component_sizes) > 1:
        largest_component_label = component_sizes[1:].argmax() + 1
        processed_mask = (labels == largest_component_label)
        processed_mask = ndimage.binary_fill_holes(processed_mask)
        return processed_mask.astype(np.uint8)
    return mask

def save_predictions_as_jpg(model, loader, device, output_dir, num_classes):
    """Generate and save predictions as comparison JPG images."""
    model.eval()
    image_counter = 0
    total_images = len(loader.dataset)
    print(f"Running inference on {total_images} images, saving to: {output_dir}")

    total_dice_before = 0.0
    total_dice_after = 0.0

    with torch.no_grad():
        for images, ground_truth_masks in loader:
            images, ground_truth_masks = images.to(device), ground_truth_masks.to(device)
            outputs = model(images)
            predicted_masks_raw = torch.argmax(outputs, dim=1)

            predicted_masks_post = torch.zeros_like(predicted_masks_raw)
            for i in range(predicted_masks_raw.shape[0]):
                predicted_masks_post[i] = torch.from_numpy(post_process_mask(predicted_masks_raw[i].cpu().numpy()))

            images_cpu = images.cpu()
            ground_truth_masks_cpu = ground_truth_masks.cpu().numpy()
            predicted_masks_raw_cpu = predicted_masks_raw.cpu().numpy()
            predicted_masks_post_cpu = predicted_masks_post.cpu().numpy()

            for i in range(images_cpu.shape[0]):
                input_image = images_cpu[i].permute(1, 2, 0).numpy()
                if input_image.shape[2] == 3:
                    input_image = input_image[:, :, 0]
                gt_mask = ground_truth_masks_cpu[i]
                pred_raw = predicted_masks_raw_cpu[i]
                pred_post = predicted_masks_post_cpu[i]

                pred_raw_tensor = torch.from_numpy(pred_raw).to(device)
                pred_post_tensor = torch.from_numpy(pred_post).to(device)
                gt_tensor = ground_truth_masks[i].to(device)
                dice_before = dice_score(pred_raw_tensor, gt_tensor, num_classes=num_classes).item()
                dice_after = dice_score(pred_post_tensor, gt_tensor, num_classes=num_classes).item()
                total_dice_before += dice_before
                total_dice_after += dice_after

                fig, ax = plt.subplots(1, 4, figsize=(24, 6))
                ax[0].imshow(input_image, cmap='gray'); ax[0].set_title("Input Image"); ax[0].axis('off')
                ax[1].imshow(gt_mask, cmap='jet', vmin=0, vmax=num_classes - 1); ax[1].set_title("Ground Truth Mask"); ax[1].axis('off')
                ax[2].imshow(pred_raw, cmap='jet', vmin=0, vmax=num_classes - 1); ax[2].set_title("Raw Prediction"); ax[2].axis('off')
                ax[3].imshow(pred_post, cmap='jet', vmin=0, vmax=num_classes - 1); ax[3].set_title("Post-Processed Prediction"); ax[3].axis('off')

                plt.tight_layout()
                filename = f"comparison_{image_counter:05d}.jpg"
                filepath = os.path.join(output_dir, filename)
                fig.savefig(filepath, format='jpg', bbox_inches='tight', pad_inches=0.1)
                plt.close(fig)

                image_counter += 1
                if image_counter % 50 == 0 or image_counter == total_images:
                    print(f" -> Saved {image_counter}/{total_images} comparison images...")

    avg_dice_before = total_dice_before / image_counter
    avg_dice_after = total_dice_after / image_counter
    return avg_dice_before, dice_after

def dice_score(pred, target, num_classes, smooth=1e-6):
    """Calculate Dice score for segmentation."""
    pred = pred.contiguous()
    target = target.contiguous()
    dice_per_class = []
    for cls in range(num_classes):
        pred_cls = (pred == cls).float()
        target_cls = (target == cls).float()
        intersection = (pred_cls * target_cls).sum()
        union = pred_cls.sum() + target_cls.sum()
        dice = (2. * intersection + smooth) / (union + smooth)
        dice_per_class.append(dice)
    return torch.mean(torch.tensor(dice_per_class))
