import argparse
import torch
from torch.utils.data import DataLoader
from model import MultiLabelModel
from dataset_prep import MultiLabelDataset
from transforms import get_train_transforms, get_val_transforms
from validate import validate
from metrics import compute_metrics


def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    model = MultiLabelModel().to(device)

    train_dataset = MultiLabelDataset(args.data, "Train", get_train_transforms())
    val_dataset   = MultiLabelDataset(args.data, "val",   get_val_transforms())

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers=2)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, num_workers=2)

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_val_loss = float("inf")

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss, preds, labels_all = validate(model, val_loader, criterion, device)
        metrics = compute_metrics(preds, labels_all)

        print(f"\nEpoch {epoch + 1}/{args.epochs}")
        print(f"  Train Loss : {total_loss / len(train_loader):.4f}")
        print(f"  Val   Loss : {val_loss:.4f}")
        print(f"  Metrics    : {metrics}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), args.save_best)
            print(f"  [INFO] Best model saved → {args.save_best}")


    torch.save(model.state_dict(), args.save_last)
    print(f"\n[INFO] Last epoch model saved → {args.save_last}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train vehicle image quality classifier")
    parser.add_argument("--data",       required=True,          help="Path to dataset root (contains Blur/ and Low_Light/)")
    parser.add_argument("--epochs",     type=int,   default=10, help="Number of training epochs")
    parser.add_argument("--batch_size", type=int,   default=32, help="Batch size")
    parser.add_argument("--lr",         type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--save_best",  default="model_best.pth", help="Path to save best checkpoint")
    parser.add_argument("--save_last",  default="model.pth",      help="Path to save last epoch checkpoint")
    args = parser.parse_args()
    train(args)