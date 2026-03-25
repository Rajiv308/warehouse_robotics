import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch
import numpy as np
import time
from src.env.warehouse_env import WarehouseEnv
from src.models.vla_model import VLAModel, freeze_language_encoder

def run_visualization(num_episodes=5, checkpoint_path="checkpoints/best_model.pth"):
    device = torch.device("cpu")  # use CPU for visualization, MPS can conflict with GUI

    # Load trained model
    print("Loading trained model...")
    model = VLAModel()
    freeze_language_encoder(model)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    print(f"Loaded checkpoint from epoch {checkpoint['epoch']+1}, val_loss={checkpoint['val_loss']:.4f}")

    # Launch environment with GUI
    print("Launching PyBullet viewer...")
    env = WarehouseEnv(render=True)
    env.initialize()

    for episode in range(num_episodes):
        obs, instruction = env.reset()
        print(f"\n--- Episode {episode+1} ---")
        print(f"Instruction: '{instruction}'")

        total_reward = 0
        best_reward = -999

        for step in range(300):  # 300 steps per episode so you can watch
            # Model predicts action from image + instruction
            img_tensor = torch.FloatTensor(obs).permute(2, 0, 1).unsqueeze(0) / 255.0
            
            with torch.no_grad():
                action = model(img_tensor, [instruction])
            
            action_np = action.squeeze().numpy()

            # Step environment
            obs, reward, done, info = env.step(action_np)
            total_reward += reward
            best_reward = max(best_reward, reward)

            # Slow it down so you can actually watch
            time.sleep(0.01)

            if step % 50 == 0:
                print(f"  Step {step:3d} | Reward: {reward:.3f} | Best so far: {best_reward:.3f}")

            if done:
                break

        print(f"Episode {episode+1} complete | Total reward: {total_reward:.2f} | Best reward: {best_reward:.3f}")
        time.sleep(1.0)  # pause between episodes

    env.close()
    print("\nVisualization complete!")

if __name__ == "__main__":
    run_visualization()
