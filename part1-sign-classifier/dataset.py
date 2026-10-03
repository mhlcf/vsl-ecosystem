import os
import json
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class VSLDataset(Dataset):
    def __init__(self, data_dir, label_map, transform=None, augment=False):
        self.data_dir = data_dir
        self.label_map = label_map
        self.transform = transform
        self.augment = augment
        self.samples = []

        label_to_idx = {name: idx for name, idx in label_map.items()}

        vocab_dirs = sorted(os.listdir(data_dir))
        for vocab_name in tqdm(vocab_dirs, desc=f"Loading {os.path.basename(data_dir)}"):
            vocab_path = os.path.join(data_dir, vocab_name)
            if not os.path.isdir(vocab_path):
                continue
            if vocab_name not in label_to_idx:
                continue
            label_idx = label_to_idx[vocab_name]
            npz_files = sorted(os.listdir(vocab_path))
            for npz_file in npz_files:
                if npz_file.endswith('.npz'):
                    self.samples.append((os.path.join(vocab_path, npz_file), label_idx))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        npz_path, label = self.samples[idx]
        data = np.load(npz_path)
        sequence = data['sequence'].astype(np.float32)
        if self.transform:
            sequence = self.transform(sequence)
        if self.augment:
            sequence = self._augment(sequence)
        return torch.from_numpy(sequence), torch.tensor(label, dtype=torch.long)

    def _augment(self, seq):
        T, D = seq.shape
        aug = seq.copy()

        if np.random.rand() < 0.3:
            noise = np.random.normal(0, 0.01, size=aug.shape).astype(np.float32)
            aug = aug + noise

        if np.random.rand() < 0.2:
            scale = np.random.uniform(0.95, 1.05)
            aug = aug * scale

        if np.random.rand() < 0.2:
            shift = np.random.normal(0, 0.01, size=D).astype(np.float32)
            aug = aug + shift

        if np.random.rand() < 0.3:
            mask_len = np.random.randint(5, 15)
            mask_start = np.random.randint(0, max(1, T - mask_len))
            aug[mask_start:mask_start + mask_len, :75] = 0

        if np.random.rand() < 0.2:
            hand_noise = np.random.normal(0, 0.02, size=(T, D - 75)).astype(np.float32)
            aug[:, 75:] = aug[:, 75:] + hand_noise

        return aug


def compute_dataset_stats(data_dir, label_map, batch_size=512, num_workers=4):
    dataset = VSLDataset(data_dir, label_map)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    all_seqs = []
    for inputs, _ in tqdm(loader, desc="Computing stats"):
        all_seqs.append(inputs.numpy())
    all_data = np.concatenate(all_seqs, axis=0)
    feat_mean = np.mean(all_data, axis=(0, 1), keepdims=True).astype(np.float32)
    feat_std = np.std(all_data, axis=(0, 1), keepdims=True).astype(np.float32) + 1e-8
    return feat_mean, feat_std


def create_dataloaders(data_root, label_map_path, batch_size=64, num_workers=4, augment=False):
    with open(label_map_path, 'r', encoding='utf-8') as f:
        label_map = json.load(f)

    train_dir = os.path.join(data_root, 'train')
    val_dir = os.path.join(data_root, 'val')
    test_dir = os.path.join(data_root, 'test')

    train_dataset = VSLDataset(train_dir, label_map, augment=augment)
    val_dataset = VSLDataset(val_dir, label_map)
    test_dataset = VSLDataset(test_dir, label_map)

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader, label_map
