# -*- coding: utf-8 -*-
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

from src.DQN import DQN

# -----------------------------
# 1. 데이터셋 클래스
# -----------------------------
class ExpertDataset(Dataset):
    def __init__(self, data_path):
        with open(data_path, "rb") as f:
            data = pickle.load(f)

        self.states = [torch.tensor(s, dtype=torch.float32) for s, _ in data]
        self.actions = [torch.tensor(a, dtype=torch.long) for _, a in data]

    def __len__(self):
        return len(self.states)

    def __getitem__(self, idx):
        return self.states[idx], self.actions[idx]

# -----------------------------
# 2. Behavior Cloning 학습 함수
# -----------------------------
def train_behavior_cloning(
    data_path="expert_data/expert_data.pkl",
    save_path="bc_model.pt",
    obs_shape=(4, 84, 84),
    num_actions=5,
    batch_size=64,
    epochs=10,
    lr=1e-4,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    dataset = ExpertDataset(data_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # DQN에서 CNN 구조만 불러와서 사용
    model = DQN(obs_shape, num_actions).network.to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        for states, actions in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            states, actions = states.to(device), actions.to(device)

            logits = model(states)              # shape: (batch, num_actions)
            loss = criterion(logits, actions)   # CrossEntropy: raw logits vs class idx

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)
        print(f"Epoch {epoch+1}: loss = {avg_loss:.4f}")

    torch.save(model.state_dict(), save_path)
    print(f"\nSaved Behavior Cloning model to {save_path}")

# -----------------------------
# 3. 실행
# -----------------------------
if __name__ == "__main__":
    train_behavior_cloning()
