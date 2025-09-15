# -*- coding: utf-8 -*-
import os
import pickle
import torch
import numpy as np
import gymnasium as gym
from tqdm import trange

from src.DQN import DQN             
from src.Preprocess import preprocess 

def collect_expert_data(
    model_path="dqn.pt",
    save_path="expert_data/expert_data.pkl",
    num_episodes=100,
    device="cuda"
):
    # CarRacing-v3 환경: discrete 모드
    env = gym.make(
        "CarRacing-v3",
        domain_randomize=False,
        continuous=False,              
        render_mode=None
    )

    obs_shape = (4, 96, 96)           
    num_actions = env.action_space.n 

    # DQN 모델 초기화 및 로드
    model = DQN(obs_shape, num_actions).network.to(device)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    expert_data = []

    for ep in trange(num_episodes, desc="Collecting expert rollouts"):
        obs, _ = env.reset(seed=ep)
        frame = preprocess(obs)
        state_stack = [frame] * 4 
        done = False

        while not done:
            state_input = np.stack(state_stack, axis=0)
            state_tensor = torch.tensor(state_input, dtype=torch.float32, device=device).unsqueeze(0)

            with torch.no_grad():
                q_values = model(state_tensor)
                action_idx = torch.argmax(q_values, dim=1).item()

            expert_data.append((state_input, action_idx))

            next_obs, reward, terminated, truncated, _ = env.step(action_idx)
            next_frame = preprocess(next_obs)
            state_stack.pop(0)
            state_stack.append(next_frame)

            done = terminated or truncated

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, "wb") as f:
        pickle.dump(expert_data, f)

    print(f"\nCollected {len(expert_data)} (state_stack, action) pairs.")
    print(f"Saved to: {save_path}")

if __name__ == "__main__":
    collect_expert_data()
