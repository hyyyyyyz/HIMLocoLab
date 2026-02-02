# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to verify exported ONNX models using HimLoco RSL-RL environment."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Verify ONNX models with HimLoco RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument(
    "--env_cfg_entry_point", type=str, default="play_env_cfg_entry_point", help="Name of the play environment configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--onnx_dir", type=str, default=None, help="Directory containing exported ONNX models.")
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import os
import time
import torch
import numpy as np

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import himlocolab.tasks  # noqa: F401
from himlocolab.rsl_rl import HimlocoVecEnvWrapper, HIMOnPolicyRunner
from himlocolab.rsl_rl.config import HIMOnPolicyRunnerCfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


class ONNXPolicy:
    """ONNX policy wrapper for dual network inference."""
    
    def __init__(self, encoder_path: str, policy_path: str, device: str = "cpu"):
        """Initialize ONNX runtime sessions.
        
        Args:
            encoder_path: Path to encoder ONNX model
            policy_path: Path to policy ONNX model
            device: Device to run inference on
        """
        import onnxruntime as ort
        
        self.device = device
        
        # Set ONNX Runtime session options
        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_EXTENDED
        
        # Load encoder
        print(f"[INFO] Loading encoder from: {encoder_path}")
        self.encoder_session = ort.InferenceSession(encoder_path, sess_options)
        self.encoder_input_name = self.encoder_session.get_inputs()[0].name
        self.encoder_output_name = self.encoder_session.get_outputs()[0].name
        encoder_input_shape = self.encoder_session.get_inputs()[0].shape
        encoder_output_shape = self.encoder_session.get_outputs()[0].shape
        print(f"[INFO] Encoder - Input: {encoder_input_shape}, Output: {encoder_output_shape}")
        
        # Load policy
        print(f"[INFO] Loading policy from: {policy_path}")
        self.policy_session = ort.InferenceSession(policy_path, sess_options)
        self.policy_input_name = self.policy_session.get_inputs()[0].name
        self.policy_output_name = self.policy_session.get_outputs()[0].name
        policy_input_shape = self.policy_session.get_inputs()[0].shape
        policy_output_shape = self.policy_session.get_outputs()[0].shape
        print(f"[INFO] Policy - Input: {policy_input_shape}, Output: {policy_output_shape}")
        
        # Store dimensions
        self.encoder_input_dim = encoder_input_shape[1]
        self.encoder_output_dim = encoder_output_shape[1]
        self.policy_output_dim = policy_output_shape[1]
        
    def __call__(self, obs: torch.Tensor) -> torch.Tensor:
        """Run inference on observations using ONNX models.
        
        Args:
            obs: Observation tensor of shape (num_envs, obs_dim)
                 Format: [history_obs (encoder_input_dim) | current_obs (remaining)]
        
        Returns:
            Action tensor of shape (num_envs, action_dim)
        """
        # Convert to numpy
        obs_np = obs.cpu().numpy().astype(np.float32)
        num_envs = obs_np.shape[0]
        
        # Split observation into history and current
        history_obs = obs_np
        current_obs = obs_np[:, :45]   # (num_envs, current_obs_dim)
        
        # Process each environment separately (ONNX models expect batch size 1)
        actions_list = []
        
        for i in range(num_envs):
            # Get single env observation
            single_history = obs_np[i:i+1]  # (1, encoder_input_dim)
            single_current = current_obs[i:i+1]  # (1, 45)
            
            # Run ONNX encoder
            encoder_output = self.encoder_session.run(
                [self.encoder_output_name],
                {self.encoder_input_name: single_history}
            )[0]  # (1, encoder_output_dim)
            
            # Concatenate current_obs with encoder output
            policy_input = np.concatenate([single_current, encoder_output], axis=1)  # (1, 64)
            
            # Run ONNX policy
            action = self.policy_session.run(
                [self.policy_output_name],
                {self.policy_input_name: policy_input}
            )[0]  # (1, action_dim)
            
            actions_list.append(action)
        
        # Stack all actions
        actions = np.concatenate(actions_list, axis=0)  # (num_envs, action_dim)
        
        # Convert back to torch tensor
        return torch.from_numpy(actions).to(self.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Verify ONNX models with HimLoco RSL-RL environment."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs

    # set the environment seed
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "himloco_rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    
    # get checkpoint path
    if args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    log_dir = os.path.dirname(resume_path)

    # determine ONNX model directory
    if args_cli.onnx_dir:
        onnx_dir = args_cli.onnx_dir
    else:
        onnx_dir = os.path.join(os.path.dirname(resume_path), "exported")
    
    encoder_path = os.path.join(onnx_dir, "encoder.onnx")
    policy_path = os.path.join(onnx_dir, "policy.onnx")
    
    # verify ONNX files exist
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder ONNX model not found: {encoder_path}")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy ONNX model not found: {policy_path}")
    
    print(f"[INFO] Using ONNX models from: {onnx_dir}")

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play_onnx"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during playback.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length
    )

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] history_length: {env.history_length}")
    print(f"[INFO] num_obs (total): {env.num_obs}")
    if env.num_one_step_privileged_obs is not None:
        print(f"[INFO] num_one_step_privileged_obs: {env.num_one_step_privileged_obs}")
        print(f"[INFO] privileged_history_length: {env.privileged_history_length}")
        print(f"[INFO] num_privileged_obs (total): {env.num_privileged_obs}")
    print(f"[INFO] num_actions: {env.num_actions}")

    # create ONNX policy
    policy = ONNXPolicy(
        encoder_path=encoder_path,
        policy_path=policy_path,
        device=env.device
    )

    print(f"[INFO] ONNX policy loaded successfully")
    print(f"[INFO] Expected observation format:")
    print(f"  - Total obs dim: {env.num_obs}")
    print(f"  - Encoder input dim: {policy.encoder_input_dim}")
    print(f"  - Current obs dim: {env.num_obs - policy.encoder_input_dim}")
    print(f"  - Encoder output dim: {policy.encoder_output_dim}")
    print(f"  - Action dim: {policy.policy_output_dim}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # simulate environment
    print("[INFO] Starting ONNX model verification playback...")
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # Run ONNX policy
            actions = policy(obs)
            
            # env stepping
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(actions)
        
        if args_cli.video:
            timestep += 1
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    env.close()
    
    print("[INFO] ONNX model verification completed successfully!")


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
