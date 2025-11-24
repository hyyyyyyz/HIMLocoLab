# Copyright (c) 2025, HimLoco Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Export utilities for HimLoco dual network architecture (encoder + policy)."""

import copy
import os
import torch
import torch.nn.functional as F


def export_himloco_policy_as_jit(
    actor_critic: object,
    path: str,
    policy_filename: str = "policy.pt",
):
    """Export HimLoco dual network (encoder + policy) as a single TorchScript JIT file.
    
    The exported model combines encoder and policy into a single forward pass:
    obs_history -> encoder -> [vel, latent] -> policy -> actions
    
    Args:
        actor_critic: The HIMActorCritic module containing estimator and actor.
        path: The path to the saving directory.
        policy_filename: The name of exported policy JIT file. Defaults to "policy.pt".
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # Export as single combined JIT model
    exporter = _HimlcoPolicyJitExporter(actor_critic)
    exporter.export(path, policy_filename)
    
    print(f"[INFO] Exported HimLoco policy JIT to: {os.path.join(path, policy_filename)}")


class _HimlcoPolicyJitExporter(torch.nn.Module):
    """Exporter for HimLoco policy as TorchScript JIT (encoder + policy combined)."""
    
    def __init__(self, actor_critic):
        super().__init__()
        # Copy actor and encoder networks
        self.actor = copy.deepcopy(actor_critic.actor)
        self.estimator = copy.deepcopy(actor_critic.estimator.encoder)
    
    def forward(self, obs_history):
        """Forward pass: obs_history -> actions
        
        Args:
            obs_history: Historical observations [batch, history_size * num_one_step_obs]
            
        Returns:
            actions: Action outputs [batch, num_actions]
        """
        # Run encoder: extract vel and latent from observation history
        parts = self.estimator(obs_history)
        
        # Split into vel and latent (assuming encoder outputs 19 dims: 3 vel + 16 latent)
        vel = parts[..., :3]
        z = parts[..., 3:19]
        
        # Normalize latent representation (L2 normalization)
        z = F.normalize(z, dim=-1, p=2.0)
        
        # Get current observation (first num_one_step_obs dims)
        # Assuming obs_history has shape [batch, history_size * 45], get first 45 dims
        current_obs = obs_history[..., :45]
        
        # Concatenate: [current_obs, vel, latent] -> policy
        policy_input = torch.cat((current_obs, vel, z), dim=-1)
        
        # Run policy network
        actions = self.actor(policy_input)
        
        return actions
    
    def export(self, path, filename):
        """Export the combined policy model to TorchScript JIT format."""
        self.to("cpu")
        self.eval()
        
        # Trace the model with dummy input
        obs_history = torch.zeros(1, self.estimator[0].in_features)
        
        # Use torch.jit.script for better compatibility
        traced_script_module = torch.jit.script(self)
        
        # Save to file
        save_path = os.path.join(path, filename)
        traced_script_module.save(save_path)
        print(f"[INFO] Successfully saved TorchScript JIT model to: {save_path}")


def export_himloco_policy_as_onnx(
    actor_critic: object,
    path: str,
    encoder_filename: str = "encoder.onnx",
    policy_filename: str = "policy.onnx",
    verbose: bool = False
):
    """Export HimLoco dual network (encoder + policy) as separate ONNX files.
    
    Args:
        actor_critic: The HIMActorCritic module containing estimator and actor.
        path: The path to the saving directory.
        encoder_filename: The name of exported encoder ONNX file. Defaults to "encoder.onnx".
        policy_filename: The name of exported policy ONNX file. Defaults to "policy.onnx".
        verbose: Whether to print the model summary. Defaults to False.
    """
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    
    # Export encoder network (obs_history -> [vel, latent])
    encoder_exporter = _HimlocoEncoderOnnxExporter(actor_critic, verbose)
    encoder_exporter.export(path, encoder_filename)
    
    # Export policy network (obs + vel + latent -> actions)
    policy_exporter = _HimlcoPolicyOnnxExporter(actor_critic, verbose)
    policy_exporter.export(path, policy_filename)
    
    print(f"[INFO] Exported encoder to: {os.path.join(path, encoder_filename)}")
    print(f"[INFO] Exported policy to: {os.path.join(path, policy_filename)}")


class _HimlocoEncoderOnnxExporter(torch.nn.Module):
    """Exporter for HimLoco encoder network (estimator)."""
    
    def __init__(self, actor_critic, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Copy encoder from estimator
        self.encoder = copy.deepcopy(actor_critic.estimator.encoder)
        
    def forward(self, obs_history):
        """Forward pass: obs_history -> [vel, latent]
        
        Args:
            obs_history: Historical observations [batch, history_size * num_one_step_obs]
            
        Returns:
            encoder_output: [vel(3) + latent(16)] concatenated output [batch, 19]
        """
        import torch.nn.functional as F
        
        # Run encoder network
        parts = self.encoder(obs_history)
        
        # Split into vel and latent
        vel = parts[..., :3]
        z = parts[..., 3:]
        
        # Normalize latent representation (L2 normalization)
        z = F.normalize(z, dim=-1, p=2)
        
        # Concatenate vel and normalized latent
        encoder_output = torch.cat([vel, z], dim=-1)
        
        return encoder_output
    
    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        
        # Create dummy input: obs_history
        obs_history = torch.zeros(1, self.encoder[0].in_features)
        
        torch.onnx.export(
            self,
            obs_history,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs_history"],
            output_names=["encoder_output"],
            dynamic_axes={},
        )


class _HimlcoPolicyOnnxExporter(torch.nn.Module):
    """Exporter for HimLoco policy network (actor)."""
    
    def __init__(self, actor_critic, verbose=False):
        super().__init__()
        self.verbose = verbose
        # Copy actor network
        self.actor = copy.deepcopy(actor_critic.actor)
        
    def forward(self, obs):
        """Forward pass: [current_obs + vel + latent] -> actions
        
        Args:
            obs: Concatenated input [batch, num_one_step_obs + 3 + 16]
                 = [current_obs(45) + vel(3) + latent(16)] = 64 dims
            
        Returns:
            actions: Action outputs [batch, num_actions]
        """
        return self.actor(obs)
    
    def export(self, path, filename):
        self.to("cpu")
        self.eval()
        
        obs = torch.zeros(1, self.actor[0].in_features)
        
        torch.onnx.export(
            self,
            obs,
            os.path.join(path, filename),
            export_params=True,
            opset_version=11,
            verbose=self.verbose,
            input_names=["obs"],
            output_names=["actions"],
            dynamic_axes={},
        )
