# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to compare PyTorch and ONNX model outputs in Isaac Lab simulation."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Compare PyTorch and ONNX models with HimLoco RSL-RL agent.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--num_steps", type=int, default=100, help="Number of steps to run comparison.")
# append HimLoco RSL-RL cli arguments
cli_args.add_himloco_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Set headless mode as default
if "--headless" not in sys.argv:
    sys.argv.append("--headless")

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
import onnxruntime as ort
import pandas as pd
import matplotlib.pyplot as plt

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict

import himlocolab.tasks  # noqa: F401
from himlocolab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
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
        """Run inference on observations.
        
        Args:
            obs: Observation tensor of shape (num_envs, obs_dim)
        
        Returns:
            Action tensor of shape (num_envs, action_dim)
        """
        # Convert to numpy
        obs_np = obs.cpu().numpy().astype(np.float32)
        num_envs = obs_np.shape[0]
        
        # The entire obs is the history input for encoder (270D)
        history_obs = obs_np  # (num_envs, encoder_input_dim)
        
        # Extract current observation from the last time step in history
        one_step_obs_dim = self.encoder_input_dim // 6  # Assuming history_length = 6
        current_obs = obs_np[:, :one_step_obs_dim]  # First 45D is current observation
        
        # Process each environment separately (ONNX models expect batch size 1)
        actions_list = []
        for i in range(num_envs):
            # Get single env observation
            single_history = history_obs[i:i+1]  # (1, encoder_input_dim) = (1, 270)
            single_current = current_obs[i:i+1]  # (1, one_step_obs_dim) = (1, 45)
            
            # Run encoder
            encoder_output = self.encoder_session.run(
                [self.encoder_output_name],
                {self.encoder_input_name: single_history}
            )[0]  # (1, encoder_output_dim) = (1, 19)
            
            # Run policy
            policy_input = np.concatenate([single_current, encoder_output], axis=1)  # (1, 64)
            action = self.policy_session.run(
                [self.policy_output_name],
                {self.policy_input_name: policy_input}
            )[0]  # (1, 12)
            
            actions_list.append(action)
        
        # Stack all actions
        actions = np.concatenate(actions_list, axis=0)  # (num_envs, action_dim)
        
        # Convert back to torch tensor
        return torch.from_numpy(actions).to(self.device)


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Compare PyTorch and ONNX models in HimLoco RSL-RL environment."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]

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

    # Determine ONNX model directory
    onnx_dir = os.path.join(os.path.dirname(resume_path), "exported")
    encoder_path = os.path.join(onnx_dir, "encoder.onnx")
    policy_path = os.path.join(onnx_dir, "policy.onnx")
    
    # Verify ONNX files exist
    if not os.path.exists(encoder_path):
        raise FileNotFoundError(f"Encoder ONNX model not found: {encoder_path}")
    if not os.path.exists(policy_path):
        raise FileNotFoundError(f"Policy ONNX model not found: {policy_path}")
    
    print(f"[INFO] Using ONNX models from: {onnx_dir}")

    # set the log directory for the environment
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap around environment for HimLoco RSL-RL
    env = HimlocoVecEnvWrapper(
        env,
        history_length=agent_cfg.history_length,
        privileged_history_length=agent_cfg.privileged_history_length
    )

    print(f"[INFO] Environment wrapped successfully")
    print(f"[INFO] num_envs: {env.num_envs}")
    print(f"[INFO] num_one_step_obs: {env.num_one_step_obs}")
    print(f"[INFO] num_obs (total): {env.num_obs}")

    print(f"[INFO] Loading model checkpoint from: {resume_path}")
    
    # create runner from HimLoco RSL-RL and load PyTorch policy
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    runner.load(resume_path)
    pt_policy = runner.get_inference_policy(device=env.device)

    # create ONNX policy
    onnx_policy = ONNXPolicy(
        encoder_path=encoder_path,
        policy_path=policy_path,
        device=env.device
    )

    print(f"[INFO] Both policies loaded successfully")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    
    # Statistics for comparison
    action_diffs = []
    estimator_diffs = []
    
    # simulate environment
    print(f"[INFO] Starting comparison for {args_cli.num_steps} steps...")
    print("[INFO] Step | PT Actions | ONNX Actions | Action Diff | PT Estimator | ONNX Estimator | Estimator Diff")
    print("-" * 130)
    
    while simulation_app.is_running() and timestep < args_cli.num_steps:
        start_time = time.time()
        
        # run everything in inference mode
        with torch.inference_mode():
            # Get PyTorch policy and estimator output
            pt_actions, pt_estimator = runner.alg.actor_critic.test_inference(obs)
            
            # Get ONNX policy output (using ONNX encoder)
            onnx_actions = onnx_policy(obs)
            
            # Extract ONNX estimator output separately
            obs_np = obs.cpu().numpy().astype(np.float32)
            onnx_estimator_list = []
            for i in range(obs_np.shape[0]):
                single_obs = obs_np[i:i+1]
                encoder_input_name = onnx_policy.encoder_session.get_inputs()[0].name
                encoder_output_name = onnx_policy.encoder_session.get_outputs()[0].name
                onnx_est = onnx_policy.encoder_session.run(
                    [encoder_output_name],
                    {encoder_input_name: single_obs}
                )[0]
                onnx_estimator_list.append(onnx_est)
            onnx_estimator = np.concatenate(onnx_estimator_list, axis=0)
            onnx_estimator = torch.from_numpy(onnx_estimator).to(obs.device)
            
            # Calculate differences
            action_diff = torch.abs(pt_actions - onnx_actions)
            estimator_diff = torch.abs(pt_estimator - onnx_estimator)
            action_diffs.append(action_diff.cpu().numpy())
            estimator_diffs.append(estimator_diff.cpu().numpy())
            
            # Print comparison for first environment (every 10 steps)
            if timestep % 10 == 0:
                pt_act = pt_actions[0].cpu().numpy()
                onnx_act = onnx_actions[0].cpu().numpy()
                pt_est = pt_estimator[0].cpu().numpy()
                onnx_est = onnx_estimator[0].cpu().numpy()
                
                print(f"{timestep:4d} | {pt_act[:3]} | {onnx_act[:3]} | {action_diff[0].max():.6f} | {pt_est[:3]} | {onnx_est[:3]} | {estimator_diff[0].max():.6f}")
            
            # Use PyTorch actions for environment step
            obs, privileged_obs, rewards, dones, infos, termination_ids, termination_privileged_obs = env.step(pt_actions)
        
        timestep += 1

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # Print comparison statistics
    print("\n" + "="*80)
    print("COMPARISON COMPLETED - Displaying results in figures...")
    print("="*80)
    
    # Generate comparison summary tables
    summary_data = []
    action_detail_rows = []
    estimator_detail_rows = []
    
    if action_diffs:
        all_action_diffs = np.concatenate(action_diffs, axis=0)
        summary_data.append({
            'Comparison': 'PT Actions vs ONNX Actions',
            'Max': all_action_diffs.max(),
            'Mean': all_action_diffs.mean(),
            'Std': all_action_diffs.std(),
            'Median': np.median(all_action_diffs),
            'P95': np.percentile(all_action_diffs, 95),
            'P99': np.percentile(all_action_diffs, 99)
        })
        # Add per-action details
        for i in range(all_action_diffs.shape[1]):
            action_detail_rows.append([
                f"Action {i}",
                f"{all_action_diffs[:, i].max():.8f}",
                f"{all_action_diffs[:, i].mean():.8f}",
                f"{all_action_diffs[:, i].std():.8f}",
                f"{np.median(all_action_diffs[:, i]):.8f}",
                f"{np.percentile(all_action_diffs[:, i], 95):.8f}",
                f"{np.percentile(all_action_diffs[:, i], 99):.8f}"
            ])
    
    if estimator_diffs:
        all_estimator_diffs = np.concatenate(estimator_diffs, axis=0)
        summary_data.append({
            'Comparison': 'PT Estimator vs ONNX Estimator',
            'Max': all_estimator_diffs.max(),
            'Mean': all_estimator_diffs.mean(),
            'Std': all_estimator_diffs.std(),
            'Median': np.median(all_estimator_diffs),
            'P95': np.percentile(all_estimator_diffs, 95),
            'P99': np.percentile(all_estimator_diffs, 99)
        })
        # Add per-output details
        for i in range(all_estimator_diffs.shape[1]):
            estimator_detail_rows.append([
                f"Output {i}",
                f"{all_estimator_diffs[:, i].max():.8f}",
                f"{all_estimator_diffs[:, i].mean():.8f}",
                f"{all_estimator_diffs[:, i].std():.8f}",
                f"{np.median(all_estimator_diffs[:, i]):.8f}",
                f"{np.percentile(all_estimator_diffs[:, i], 95):.8f}",
                f"{np.percentile(all_estimator_diffs[:, i], 99):.8f}"
            ])
    
    # Figure 1: Summary comparison table
    fig1, ax1 = plt.subplots(figsize=(14, 3))
    ax1.axis('tight')
    ax1.axis('off')
    
    df_summary = pd.DataFrame(summary_data)
    summary_table_data = [df_summary.columns.tolist()]
    for idx, row in df_summary.iterrows():
        summary_table_data.append([
            row['Comparison'],
            f"{row['Max']:.8f}",
            f"{row['Mean']:.8f}",
            f"{row['Std']:.8f}",
            f"{row['Median']:.8f}",
            f"{row['P95']:.8f}",
            f"{row['P99']:.8f}"
        ])
    
    table1 = ax1.table(cellText=summary_table_data, cellLoc='center', loc='center',
                       colWidths=[0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table1.auto_set_font_size(False)
    table1.set_fontsize(9)
    table1.scale(1, 2)
    
    for i in range(len(summary_table_data[0])):
        table1[(0, i)].set_facecolor('#40466e')
        table1[(0, i)].set_text_props(weight='bold', color='white')
    
    for i in range(1, len(summary_table_data)):
        for j in range(len(summary_table_data[0])):
            table1[(i, j)].set_facecolor('#e8f4f8')
    
    fig1.suptitle('Comparison Summary Statistics', fontsize=14, fontweight='bold')
    plt.show()
    plt.close()
    
    # Figure 2: Action details
    if action_detail_rows:
        fig2, ax2 = plt.subplots(figsize=(14, max(4, len(action_detail_rows) * 0.4)))
        ax2.axis('tight')
        ax2.axis('off')
        
        action_table_data = [['Element', 'Max', 'Mean', 'Std', 'Median', 'P95', 'P99']]
        action_table_data.extend(action_detail_rows)
        
        table2 = ax2.table(cellText=action_table_data, cellLoc='center', loc='center',
                           colWidths=[0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        table2.auto_set_font_size(False)
        table2.set_fontsize(9)
        table2.scale(1, 1.8)
        
        for i in range(len(action_table_data[0])):
            table2[(0, i)].set_facecolor('#40466e')
            table2[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(action_table_data)):
            for j in range(len(action_table_data[0])):
                if i % 2 == 0:
                    table2[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table2[(i, j)].set_facecolor('#ffffff')
        
        fig2.suptitle('PT Actions vs ONNX Actions - Per-Action Breakdown', fontsize=14, fontweight='bold')
        plt.show()
        plt.close()
    
    # Figure 3: Estimator details
    if estimator_detail_rows:
        fig3, ax3 = plt.subplots(figsize=(14, max(4, len(estimator_detail_rows) * 0.4)))
        ax3.axis('tight')
        ax3.axis('off')
        
        estimator_table_data = [['Element', 'Max', 'Mean', 'Std', 'Median', 'P95', 'P99']]
        estimator_table_data.extend(estimator_detail_rows)
        
        table3 = ax3.table(cellText=estimator_table_data, cellLoc='center', loc='center',
                           colWidths=[0.3, 0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
        table3.auto_set_font_size(False)
        table3.set_fontsize(9)
        table3.scale(1, 1.8)
        
        for i in range(len(estimator_table_data[0])):
            table3[(0, i)].set_facecolor('#40466e')
            table3[(0, i)].set_text_props(weight='bold', color='white')
        
        for i in range(1, len(estimator_table_data)):
            for j in range(len(estimator_table_data[0])):
                if i % 2 == 0:
                    table3[(i, j)].set_facecolor('#f0f0f0')
                else:
                    table3[(i, j)].set_facecolor('#ffffff')
        
        fig3.suptitle('PT Estimator vs ONNX Estimator - Per-Output Breakdown', fontsize=14, fontweight='bold')
        plt.show()
        plt.close()
    
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
