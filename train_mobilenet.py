"""Train the MobileNet binary classifier (clean vs contaminated).

Stage two of the solar-sweeper pipeline. Uses the leakage-free binary dataset
under data/binary_classification (train/val/test, classes clean=0, contaminated=1).
Transfer-learns from an ImageNet-pretrained MobileNetV3-Small and swaps in a
2-way head.
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import models, transforms
from torchvision.datasets import ImageFolder

TB_DIR = r"C:/Users/VANSH/runs/detect/runs/train/mobilenet_binary"

DATA = "data/binary_classification"
EPOCHS = 20
BATCH = 32
LR = 3e-4
IMG = 224
PATIENCE = 6

IMNET_MEAN = [0.485, 0.456, 0.406]
IMNET_STD = [0.229, 0.224, 0.225]

train_tf = transforms.Compose([
    transforms.Resize((IMG + 32, IMG + 32)),
    transforms.RandomCrop(IMG),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])
eval_tf = transforms.Compose([
    transforms.Resize((IMG, IMG)),
    transforms.ToTensor(),
    transforms.Normalize(IMNET_MEAN, IMNET_STD),
])


def build_model(num_classes=2):
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
    in_feats = model.classifier[-1].in_features
    model.classifier[-1] = nn.Linear(in_feats, num_classes)
    return model


def binary_metrics(y_true, y_pred, y_prob):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    y_prob = np.asarray(y_prob)
    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    acc = (tp + tn) / max(tp + fp + fn + tn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    # ROC-AUC via rank statistic (Mann-Whitney U)
    pos = y_prob[y_true == 1]
    neg = y_prob[y_true == 0]
    if len(pos) and len(neg):
        order = np.argsort(np.concatenate([pos, neg]))
        ranks = np.empty(len(order), dtype=float)
        ranks[order] = np.arange(1, len(order) + 1)
        auc = (ranks[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg))
    else:
        auc = float("nan")
    return acc, prec, rec, auc, np.array([[tn, fp], [fn, tp]])


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    train_ds = ImageFolder(f"{DATA}/train", transform=train_tf)
    val_ds = ImageFolder(f"{DATA}/val", transform=eval_tf)
    test_ds = ImageFolder(f"{DATA}/test", transform=eval_tf)
    print("classes:", train_ds.classes)

    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True, num_workers=4, pin_memory=True)
    val_dl = DataLoader(val_ds, batch_size=BATCH, num_workers=4, pin_memory=True)
    test_dl = DataLoader(test_ds, batch_size=BATCH, num_workers=4, pin_memory=True)

    model = build_model().to(device)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
    writer = SummaryWriter(TB_DIR)

    best_val = -1.0
    best_state = None
    bad = 0
    for epoch in range(1, EPOCHS + 1):
        model.train()
        run_loss = 0.0
        n_batches = 0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            run_loss += loss.item()
            n_batches += 1
        sched.step()
        train_loss = run_loss / max(n_batches, 1)
        writer.add_scalar("train_loss", train_loss, epoch)

        model.eval()
        correct = total = 0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                pred = model(x).argmax(1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        val_acc = correct / total
        writer.add_scalar("val_acc", val_acc, epoch)
        writer.add_scalar("lr", opt.param_groups[0]["lr"], epoch)
        print(f"epoch {epoch:2}: train_loss {train_loss:.4f}  val_acc {val_acc:.4f}")

        if val_acc > best_val:
            best_val = val_acc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            torch.save(best_state, "mobilenet_binary_best.pth")
            bad = 0
        else:
            bad += 1
            if bad >= PATIENCE:
                print("early stopping at epoch", epoch)
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "mobilenet_binary_best.pth")

    # test evaluation
    model.eval()
    ys, ps, probs = [], [], []
    with torch.no_grad():
        for x, y in test_dl:
            x = x.to(device)
            logits = model(x)
            ys += y.tolist()
            ps += logits.argmax(1).tolist()
            probs += torch.softmax(logits, 1)[:, 1].tolist()
    acc, prec, rec, auc, cm = binary_metrics(ys, ps, probs)
    writer.add_scalar("test_acc", acc, 0)
    writer.add_scalar("test_precision", prec, 0)
    writer.add_scalar("test_recall", rec, 0)
    writer.add_scalar("test_roc_auc", auc, 0)
    writer.close()
    print("\n--- Test results ---")
    print(f"accuracy : {acc:.4f}")
    print(f"precision: {prec:.4f}")
    print(f"recall   : {rec:.4f}")
    print(f"roc_auc  : {auc:.4f}")
    print("confusion (rows=true, cols=pred) [[tn,fp],[fn,tp]]:")
    print(cm)


if __name__ == "__main__":
    main()
