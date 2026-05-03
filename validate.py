import torch


def validate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    total_loss = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            probs = torch.sigmoid(outputs)
            preds = (probs > threshold).float()

            all_preds.append(preds.cpu())
            all_labels.append(labels.cpu())

            total_loss += loss.item()

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)

    return total_loss / len(loader), all_preds, all_labels