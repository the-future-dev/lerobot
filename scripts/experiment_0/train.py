#!/usr/bin/env python3
"""
Train a Diffusion Policy model on the PushT environment using keypoints.
"""
import logging
import argparse
from pathlib import Path

import torch
import wandb

from lerobot.common.utils.utils import init_logging
from lerobot.configs.train import TrainPipelineConfig
from lerobot.configs.default import DatasetConfig, WandBConfig, EvalConfig
from lerobot.configs.types import PolicyFeature, FeatureType, NormalizationMode
from lerobot.common.policies.diffusion.configuration_diffusion import DiffusionConfig
from lerobot.common.policies.diffusion.modeling_diffusion import DiffusionPolicy
from lerobot.scripts.train import train
from lerobot.common.envs.factory import make_env_config

import os

def get_wandb_key():
    WAND_TOKEN = os.getenv("WAND_TOKEN")
    if not WAND_TOKEN:
        raise ValueError(f"WAND_TOKEN not found")
    return WAND_TOKEN

def parse_args():
    parser = argparse.ArgumentParser(description="Train a diffusion policy on PushT with keypoints")
    parser.add_argument(
        "--job-name", 
        type=str, 
        help="Name of the job in Wandb"
    )
    parser.add_argument(
        "--env-type", 
        type=str, 
        help="PushT Env Type"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="Directory to save outputs"
    )
    parser.add_argument(
        "--seed", 
        type=int, 
        default=42, 
        help="Random seed"
    )
    parser.add_argument(
        "--dataset-repo-id", 
        type=str, 
        help="Repository ID for pushing to Hub"
    )
    parser.add_argument(
        "--push-to-hub", 
        action="store_true",
        help="Push trained model to Hugging Face Hub"
    )
    parser.add_argument(
        "--hub-repo-id", 
        type=str, 
        help="Repository ID for pushing to Hub"
    )
    return parser.parse_args()

def main():
    args = parse_args()
    init_logging()
    wandb.login(key=get_wandb_key())
    logging.info("Starting training for Diffusion Policy with keypoints")

    # Define device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Using device: {device}")
    output_dir = Path(args.output_dir)

    # Create environment config for evaluation
    env_config = make_env_config(
        env_type="pusht",
        obs_type=args.env_type,
    )

    # Define policy input and output features
    input_features = {
        "observation.state": PolicyFeature(type=FeatureType.STATE, shape=(2,)),
    }

    if args.env_type in ["environment_diff_state", "environment_state_agent_pos"]:
        input_features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(16,))
    elif args.env_type == "environment_state_agent_pos_privileged":
        input_features["observation.environment_state"] = PolicyFeature(type=FeatureType.ENV, shape=(32,))
    elif args.env_type == "pixels_agent_pos":
        input_features["pixels"] = PolicyFeature(type=FeatureType.VISUAL, shape=(96, 96, 3))

    output_features = {
        "action": PolicyFeature(type=FeatureType.ACTION, shape=(2,))
    }

    # Create normalization mapping
    normalization_mapping = {
        "STATE": NormalizationMode.MIN_MAX,
        "ENV": NormalizationMode.MIN_MAX,
        "ACTION": NormalizationMode.MIN_MAX,
        "VISUAL": NormalizationMode.IDENTITY,
    }

    encoder_vision_config = {}
    encoder_states_config = {}

    # Vision backbone.
    if args.env_type == "pixels_agent_pos":
        encoder_vision_config = {
            "vision_backbone": "resnet18",
            "crop_shape": (76, 76),
            "crop_is_random": True,
            "pretrained_backbone_weights": None,
            "use_group_norm": True,
            # "spatial_softmax_num_keypoints": 32,
            "use_separate_rgb_encoder_per_camera": True,
        }
    
    # Architecture / modeling.
    # State encoder parameters
    if args.env_type in ["environment_diff_state", "environment_state_agent_pos", "environment_state_agent_pos_privileged"]:
        encoder_states_config = {
            "state_backbone": "MLP",
            "state_encoder_block_channels": [64, 128],
            "state_encoder_feature_dim": 256,
            "state_encoder_use_layernorm": True,
        }
    
    # Create the policy configuration
    policy_config = DiffusionConfig(
        # Inputs / output structure.
        n_obs_steps=2,
        horizon=16,
        n_action_steps=8,
        input_features=input_features,
        output_features=output_features,
        normalization_mapping=normalization_mapping,
        
        **encoder_vision_config,
        **encoder_states_config,

        # Unet.
        down_dims=(512, 1024, 2048),
        kernel_size=5,
        n_groups=8,
        
        # Noise scheduler.
        noise_scheduler_type="DDPM",
        num_train_timesteps=100,
        beta_start=0.0001,
        beta_end=0.02,
        beta_schedule="squaredcos_cap_v2",
        variance_type="fixed_small",
        clip_sample=True,
        # clip_sample_range=1.0,
        prediction_type="epsilon",
        
        # Training hyperparameters
        optimizer_type="adamw",
        optimizer_lr=0.0001,
        optimizer_betas=(0.95, 0.999),
        optimizer_eps=1e-8,
        optimizer_weight_decay=1e-6,
        # scheduler_name="cosine",
        scheduler_warmup_steps=500,
        device=device,

        # ??? ema:
        #     _target_: diffusion_policy.model.diffusion.ema_model.EMAModel
        #     update_after_step: 0
        #     inv_gamma: 1.0
        #     power: 0.75
        #     min_value: 0.0
        #     max_value: 0.9999
    )

    # Create dataset config
    dataset_config = DatasetConfig(
        repo_id=args.dataset_repo_id
    )

    # Create WandB config
    wandb_config = WandBConfig(
        enable=True,
        project="diffusion-pusht-keypoints",
        entity="fiatlux",
    )

    # Create eval config
    eval_config = EvalConfig(
        n_episodes=50,
        batch_size=10,
    )

    # Create training pipeline config
    train_batch_size = 64
    num_workers = 8
    epochs = 200000
    train_eval_freq = 2500
    train_log_freq = 100
    train_save_freq = 10000
    train_config = TrainPipelineConfig(
        dataset=dataset_config,
        env=env_config,
        policy=policy_config,
        output_dir=output_dir,
        job_name=args.job_name,
        seed=args.seed,
        num_workers=num_workers,
        batch_size=train_batch_size,
        steps=epochs,
        eval_freq=train_eval_freq,
        log_freq=train_log_freq,
        save_checkpoint=True,
        save_freq=train_save_freq,
        use_policy_training_preset=True,
        eval=eval_config,
        wandb=wandb_config,
    )

    logging.info("Starting training with config")
    train(train_config)
    
    # Push to Hugging Face Hub if requested
    if args.push_to_hub:
        logging.info(f"Pushing trained model to {args.hub_repo_id}")
        last_checkpoint = f"{output_dir}/checkpoints/last/pretrained_model"
        try:
            trained_policy = DiffusionPolicy.from_pretrained(
                last_checkpoint, 
                config=policy_config
            )
            trained_policy.push_to_hub(
                args.hub_repo_id,
                commit_message="training complete :>",
                private=False
            )
            logging.info("Successfully pushed model to Hub")
        except Exception as e:
            logging.error(f"Failed to push model to Hub: {e}")
        finally:
            # Clean up to avoid memory issues
            if 'trained_policy' in locals():
                del trained_policy
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main() 