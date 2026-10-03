import os
import json
import sys
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm

from dataset import create_dataloaders
from model import VSLModel


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
LABEL_MAP_PATH = os.path.join(ROOT_DIR, 'label_map.json')
LOG_DIR = os.path.join(ROOT_DIR, 'logs')
CHECKPOINT_DIR = os.path.join(ROOT_DIR, 'checkpoint')
BEST_MODEL_PATH = os.path.join(ROOT_DIR, 'best_model.pth')
SCALER_PATH = os.path.join(ROOT_DIR, 'scaler.npz')
LOG_FILE = os.path.join(ROOT_DIR, 'training_output.log')

BATCH_SIZE = 64
NUM_EPOCHS = 150
LEARNING_RATE = 0.001
WEIGHT_DECAY = 5e-4
NUM_WORKERS = 4
PATIENCE = 20
CLIP_GRAD_NORM = 3.0

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class Logger:
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, 'a', encoding='utf-8')

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()


print(f"Using device: {device}")
if device.type == 'cuda':
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")


def compute_normalization_stats(loader, device):
    num_batches = len(loader)
    if num_batches == 0:
        return None, None

    all_means = []
    all_stds = []
    for inputs, _ in tqdm(loader, desc="Computing stats"):
        batch = inputs.numpy()
        batch_flat = batch.reshape(-1, batch.shape[-1])
        all_means.append(np.mean(batch_flat, axis=0, keepdims=True))
        all_stds.append(np.std(batch_flat, axis=0, keepdims=True) + 1e-8)

    concat_means = np.concatenate(all_means, axis=0)
    concat_stds = np.concatenate(all_stds, axis=0)

    global_mean = np.mean(concat_means, axis=0, keepdims=True).astype(np.float32)
    global_std = np.mean(concat_stds, axis=0, keepdims=True).astype(np.float32)

    scaler_mean = torch.from_numpy(global_mean.reshape(1, 1, -1)).float().to(device)
    scaler_std = torch.from_numpy(global_std.reshape(1, 1, -1)).float().to(device)
    return scaler_mean, scaler_std


def train_epoch(model, loader, criterion, optimizer, scaler_mean, scaler_std, device):
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training")
    for inputs, targets in pbar:
        inputs, targets = inputs.to(device), targets.to(device)
        if scaler_mean is not None:
            inputs = (inputs - scaler_mean) / scaler_std
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=CLIP_GRAD_NORM)
        optimizer.step()
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
        pbar.set_postfix(loss=loss.item(), acc=correct / max(total, 1))
    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


@torch.no_grad()
def eval_epoch(model, loader, criterion, scaler_mean, scaler_std, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    for inputs, targets in tqdm(loader, desc="Evaluating"):
        inputs, targets = inputs.to(device), targets.to(device)
        if scaler_mean is not None:
            inputs = (inputs - scaler_mean) / scaler_std
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        total_loss += loss.item() * inputs.size(0)
        _, preds = torch.max(outputs, 1)
        correct += (preds == targets).sum().item()
        total += targets.size(0)
    avg_loss = total_loss / max(total, 1)
    avg_acc = correct / max(total, 1)
    return avg_loss, avg_acc


def save_checkpoint(epoch, model, optimizer, scheduler, best_val_acc, patience_counter, history, filepath):
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'best_val_acc': best_val_acc,
        'patience_counter': patience_counter,
        'history': history,
    }, filepath)


def load_checkpoint(filepath, model, optimizer=None, scheduler=None):
    checkpoint = torch.load(filepath, map_location='cpu', weights_only=True)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer and 'optimizer_state_dict' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if scheduler and 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    return checkpoint


def main():
    sys.stdout = Logger(LOG_FILE)

    print("Loading data...")
    train_loader, val_loader, _, label_map = create_dataloaders(
        ROOT_DIR, LABEL_MAP_PATH, BATCH_SIZE, NUM_WORKERS, augment=True
    )
    num_classes = len(label_map)
    print(f"Train samples: {len(train_loader.dataset)}")
    print(f"Val samples: {len(val_loader.dataset)}")
    print(f"Number of classes: {num_classes}")

    checkpoint_path = os.path.join(CHECKPOINT_DIR, 'last.pth')
    resume_epoch = 0

    if os.path.exists(SCALER_PATH):
        print("Loading existing scaler...")
        scaler_data = np.load(SCALER_PATH)
        scaler_mean = torch.from_numpy(scaler_data['mean']).float().to(device)
        scaler_std = torch.from_numpy(scaler_data['std']).float().to(device)
        print(f"Scaler loaded from {SCALER_PATH}")
    else:
        print("Computing normalization stats from training data...")
        train_dataset = train_loader.dataset
        loader_for_stats = torch.utils.data.DataLoader(
            train_dataset, batch_size=BATCH_SIZE * 2, shuffle=False, num_workers=NUM_WORKERS
        )
        scaler_mean, scaler_std = compute_normalization_stats(loader_for_stats, device)
        np.savez(SCALER_PATH,
                 mean=scaler_mean.cpu().numpy(),
                 std=scaler_std.cpu().numpy())
        print(f"Scaler saved to {SCALER_PATH}")

    model = VSLModel(num_classes=num_classes).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {total_params:,}")

    criterion = nn.CrossEntropyLoss(label_smoothing=0.2)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    best_val_acc = 0.0
    patience_counter = 0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}

    if os.path.exists(checkpoint_path):
        print(f"Resuming from checkpoint: {checkpoint_path}")
        ckpt = load_checkpoint(checkpoint_path, model, optimizer, scheduler)
        resume_epoch = ckpt['epoch']
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        patience_counter = ckpt.get('patience_counter', 0)
        history = ckpt.get('history', history)
        print(f"Resumed at epoch {resume_epoch}, best_val_acc={best_val_acc:.4f}")

    if os.path.exists(BEST_MODEL_PATH) and resume_epoch == 0:
        print("Loading existing best_model.pth weights...")
        model.load_state_dict(torch.load(BEST_MODEL_PATH, map_location='cpu', weights_only=True))

    model = model.to(device)

    for epoch in range(resume_epoch + 1, NUM_EPOCHS + 1):
        epoch_start = time.time()
        print(f"\n{'='*60}")
        print(f"Epoch {epoch}/{NUM_EPOCHS}")

        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer,
                                            scaler_mean, scaler_std, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion,
                                       scaler_mean, scaler_std, device)

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]['lr']

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        epoch_time = time.time() - epoch_start
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val   Loss: {val_loss:.4f} | Val   Acc: {val_acc:.4f}")
        print(f"LR: {current_lr:.6f} | Time: {epoch_time:.1f}s")

        save_checkpoint(epoch, model, optimizer, scheduler,
                        best_val_acc, patience_counter, history, checkpoint_path)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), BEST_MODEL_PATH)
            print(f"*** New best model saved! Val Acc: {val_acc:.4f} ***")
        else:
            patience_counter += 1
            print(f"Patience: {patience_counter}/{PATIENCE}")

        if patience_counter >= PATIENCE:
            print(f"Early stopping triggered after {epoch} epochs.")
            break

    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].plot(history['train_loss'], label='Train Loss')
    axes[0].plot(history['val_loss'], label='Val Loss')
    axes[0].set_title('Loss')
    axes[0].legend()
    axes[1].plot(history['train_acc'], label='Train Acc')
    axes[1].plot(history['val_acc'], label='Val Acc')
    axes[1].set_title('Accuracy')
    axes[1].legend()
    plt.savefig(os.path.join(LOG_DIR, 'training_curves.png'))
    print(f"Training curves saved to {LOG_DIR}/training_curves.png")


if __name__ == '__main__':
    main()
