import torch
import time

def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, device, model_save_path, num_classes):
    """Train the model and validate it, saving the best model based on validation DSC."""
    print("\n--- Starting Training & Validation ---")
    start_time = time.time()
    history = {'train_loss': [], 'val_loss': [], 'val_dsc': []}
    best_val_dsc = 0.0

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device, dtype=torch.long)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)
        history['train_loss'].append(train_loss)

        model.eval()
        val_running_loss = 0.0
        val_running_dsc = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device, dtype=torch.long)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_running_loss += loss.item()
                predicted_masks = torch.argmax(outputs, dim=1)
                dsc = dice_score(predicted_masks, masks, num_classes)
                val_running_dsc += dsc.item()
        val_loss = val_running_loss / len(val_loader)
        val_dsc = val_running_dsc / len(val_loader)
        history['val_loss'].append(val_loss)
        history['val_dsc'].append(val_dsc)

        print(f"Epoch [{epoch+1}/{num_epochs}] | "
              f"Train Loss: {train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val DSC: {val_dsc:.4f}")

        if val_dsc > best_val_dsc:
            best_val_dsc = val_dsc
            torch.save(model.state_dict(), model_save_path)
            print(f"  -> 🎉 New best model saved! Validation DSC: {best_val_dsc:.4f}")

    end_time = time.time()
    print(f"\n--- Training Finished ---")
    print(f"Total training time: {end_time - start_time:.2f} seconds")
    return history

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