# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Interactive play script with keyboard control for HimLoco RSL-RL agent."""

"""Launch Isaac Sim Simulator first."""

import argparse
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Interactive play with HimLoco RSL-RL agent (keyboard control).")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during playback.")
parser.add_argument("--video_length", type=int, default=500, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="himloco_rsl_rl_cfg", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
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
from collections import deque

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg

import himlocolab.tasks  # noqa: F401
from himlocolab.rsl_rl import HIMOnPolicyRunner, HimlocoVecEnvWrapper
from himlocolab.rsl_rl.config import HIMOnPolicyRunnerCfg
from himlocolab.utils import export_deploy_cfg
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: HIMOnPolicyRunnerCfg):
    """Interactive play with HimLoco RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    train_task_name = task_name.replace("-Play", "")

    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_himloco_rsl_rl_cfg(agent_cfg, args_cli)
    

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

    # set the log directory for the environment
    env_cfg.log_dir = log_dir
    
    # Disable timeout termination for interactive play
    env_cfg.terminations.time_out = None

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "interactive"),
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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    
    # create runner from HimLoco RSL-RL
    runner = HIMOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    
    # load the checkpoint
    runner.load(resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.device)

    # Initialize Se2Keyboard for real-time keyboard control
    # Get command ranges from environment config if available
    try:
        if hasattr(env_cfg.commands, "base_velocity"):
            if hasattr(env_cfg.commands.base_velocity, "limit_ranges"):
                ranges = env_cfg.commands.base_velocity.limit_ranges
            else:
                ranges = env_cfg.commands.base_velocity.ranges
            
            max_lin_vel_x = ranges.lin_vel_x[1]
            max_lin_vel_y = ranges.lin_vel_y[1]
            max_ang_vel_z = ranges.ang_vel_z[1]
        else:
            max_lin_vel_x = 2.0
            max_lin_vel_y = 1.0
            max_ang_vel_z = 2.0
    except Exception as e:
        print(f"[WARNING] Could not read command ranges from config: {e}")
        max_lin_vel_x = 2.0
        max_lin_vel_y = 1.0
        max_ang_vel_z = 2.0
    
    # Create Se2Keyboard controller
    keyboard_cfg = Se2KeyboardCfg(
        sim_device=env.device,
        v_x_sensitivity=max_lin_vel_x,
        v_y_sensitivity=max_lin_vel_y,
        omega_z_sensitivity=max_ang_vel_z
    )
    keyboard = Se2Keyboard(keyboard_cfg)
    
    print(keyboard)
    print(f"[INFO] Command ranges: lin_vel_x=[{-max_lin_vel_x:.1f}, {max_lin_vel_x:.1f}], "
          f"lin_vel_y=[{-max_lin_vel_y:.1f}, {max_lin_vel_y:.1f}], "
          f"ang_vel_z=[{-max_ang_vel_z:.1f}, {max_ang_vel_z:.1f}]")

    dt = env.unwrapped.step_dt
    timestep = 0
    
    # simulate environment
    print("[INFO] Starting interactive play with real-time keyboard control...")
    obs = env.get_observations()
    
    while simulation_app.is_running():
        start_time = time.time()
        
        # Get keyboard commands from Se2Keyboard
        command = keyboard.advance()
        
        # Set command to environment through command manager
        try:
            vel_command_term = env.unwrapped.command_manager.get_term("base_velocity")
            if command.dim() == 1:
                command = command.unsqueeze(0)
            vel_command_term.vel_command_b[:] = command.to(device=env.device)
        except Exception as e:
            print(f"[WARNING] Failed to set keyboard command: {e}")
        
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
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


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
