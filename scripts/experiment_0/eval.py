import argparse

from pathlib import Path
import json
import numpy as np
import torch
import imageio
import gym_pusht  # noqa: F401
import gymnasium as gym
import matplotlib.pyplot as plt

# Import the eval_policy function from the lerobot scripts
from lerobot.scripts.eval import eval_policy
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.common.envs.factory import make_env, make_env_config

import os

def get_wandb_key():    
    WAND_TOKEN = os.getenv("WAND_TOKEN")
    if not WAND_TOKEN:
        raise ValueError(f"WAND_TOKEN not found")
    return WAND_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion policy on PushT with keypoints")
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--pretrained-path", 
        type=str,
        help="Random seed"
    )
    parser.add_argument(
        "env-obs-type",
        type=str,
        help="Type of input / output expected from the simulation gym environment"
    )
    parser.add_argument(
        "--n-envs", 
        type=int, 
        default=1, 
        help="Number of Envs"
    )
    parser.add_argument(
        "--n-episodes", 
        type=int, 
        default=1, 
        help="Number of Episodes"
    )
    parser.add_argument(
        "--n-videos", 
        type=int, 
        default=1, 
        help="Number of Videos"
    )
    
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    
    return parser.parse_args()


args = parse_args()

output_directory = Path("")
output_directory.mkdir(parents=True, exist_ok=True)
print(f"OUTPUT DIR: {output_directory}")

videos_dir = output_directory / "videos"
videos_dir.mkdir(parents=True, exist_ok=True)

device = "cuda"
pretrained_policy_path = args.pretrained_path
print(f"POLICY DIR: {pretrained_policy_path}")

policy = DiffusionPolicy.from_pretrained(pretrained_policy_path)
policy.to(device)

# print(policy) 
print("\n=== Policy Configuration ===")
print(f"Input features: {policy.config.input_features}")
print(f"Output features: {policy.config.output_features}")
print(f"Image features: {policy.config.image_features}")
print(f"Observation steps: {policy.config.n_obs_steps}")
print(f"Action steps: {policy.config.n_action_steps}")
print(f"Horizon: {policy.config.horizon}")

n_envs = args.n_envs             # Number of parallel environments
n_episodes = args.n_episodes        # Total number of episodes to evaluate
start_seed = args.seed    # Starting seed

# Create a vectorized environment
env_config = make_env_config(
    env_type="pusht",
    obs_type=args.env_obs_type,
)

# Create the vectorized environment
env = make_env(
    env_config, 
    n_envs=n_envs,
    use_async_envs=True
)

print(f"Created vectorized environment with {n_envs} parallel environments")
print(f"Running evaluation for {n_episodes} episodes starting from seed {start_seed}")


eval_results = eval_policy(
    env=env,
    policy=policy,
    n_episodes=n_episodes,
    max_episodes_rendered=args.n_videos,  # Only render 1 video
    videos_dir=videos_dir,
    return_episode_data=False,
    start_seed=start_seed,
)

# Close the environment
env.close()

# Print the aggregated metrics
print("\n=== Evaluation Results ===")
print(f"Average sum reward: {eval_results['aggregated']['avg_sum_reward']:.4f}")
print(f"Average max reward: {eval_results['aggregated']['avg_max_reward']:.4f}")
print(f"Success rate: {eval_results['aggregated']['pc_success']:.2f}%")
print(f"Total evaluation time: {eval_results['aggregated']['eval_s']:.2f} seconds")
print(f"Average time per episode: {eval_results['aggregated']['eval_ep_s']:.2f} seconds")

# If you want to analyze per-episode results
success_by_episode = [ep["success"] for ep in eval_results["per_episode"]]
rewards_by_episode = [ep["sum_reward"] for ep in eval_results["per_episode"]]

plt.figure(figsize=(12, 5))

# Plot rewards
plt.subplot(1, 2, 1)
plt.plot(rewards_by_episode)
plt.title('Rewards by Episode')
plt.xlabel('Episode')
plt.ylabel('Sum Reward')

# Plot success
plt.subplot(1, 2, 2)
plt.plot([int(s) for s in success_by_episode])
plt.title('Success by Episode')
plt.xlabel('Episode')
plt.ylabel('Success Rate')
plt.ylim(-0.1, 1.1)

plt.tight_layout()
plt.savefig(output_directory / "evaluation_results.png")
plt.show()

# Save the evaluation results to a file
with open(output_directory / "eval_results.json", "w") as f:
    json.dump(eval_results, f, indent=2)

# Print video path if any videos were generated
if "video_paths" in eval_results and eval_results["video_paths"]:
    print(f"\nGenerated video is available at: {eval_results['video_paths'][0]}")
else:
    print("No videos generated")