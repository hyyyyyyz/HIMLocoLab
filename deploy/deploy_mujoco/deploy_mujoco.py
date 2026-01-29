import time

import mujoco.viewer
import mujoco
import numpy as np
import torch
import yaml

import os

# 项目根目录
HIMLOCO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

def get_gravity_orientation(quaternion):
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)

    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD 控制计算"""
    return (target_q - q) * kp + (target_dq - dq) * kd


if __name__ == "__main__":

    config_file = f"{HIMLOCO_ROOT_DIR}/deploy/deploy_mujoco/config/go2.yaml"
    with open(config_file, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{HIMLOCO_ROOT_DIR}", HIMLOCO_ROOT_DIR)
        xml_path = config["xml_path"].replace("{HIMLOCO_ROOT_DIR}", HIMLOCO_ROOT_DIR)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_angles = np.array(config["default_angles"], dtype=np.float32)

        ang_vel_scale = config["ang_vel_scale"]
        dof_pos_scale = config["dof_pos_scale"]
        dof_vel_scale = config["dof_vel_scale"]
        action_scale = config["action_scale"]
        cmd_scale = np.array(config["cmd_scale"], dtype=np.float32)

        num_actions = config["num_actions"]
        num_obs = config["num_obs"]
        
        cmd = np.array(config["cmd_init"], dtype=np.float32)
        max_cmd = np.array(config["max_cmd"], dtype=np.float32)

    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_angles.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # 加载机器人模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    m.opt.timestep = simulation_dt

    # 加载策略
    policy = torch.jit.load(policy_path)

    with mujoco.viewer.launch_passive(m, d) as viewer:
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, d.qpos[7:], kps, np.zeros_like(kds), d.qvel[6:], kds)
            d.ctrl[:] = tau
            # 可以在 mj_step 之前或之后应用控制信号
            mujoco.mj_step(m, d)

            counter += 1
            if counter % control_decimation == 0:
                # 应用控制信号

                # 建立观测
                qj = d.qpos[7:]
                dqj = d.qvel[6:]
                quat = d.qpos[3:7]
                omega = d.qvel[3:6]

                qj = (qj - default_angles) * dof_pos_scale
                dqj = dqj * dof_vel_scale
                gravity_orientation = get_gravity_orientation(quat)
                omega = omega * ang_vel_scale

                obs[:3] = cmd * cmd_scale * max_cmd
                obs[3:6] = omega
                obs[6:9] = gravity_orientation
                obs[9 : 9 + num_actions] = qj
                obs[9 + num_actions : 9 + 2 * num_actions] = dqj
                obs[9 + 2 * num_actions : 9 + 3 * num_actions] = action
                obs_tensor = torch.from_numpy(obs).unsqueeze(0)
                # 策略推理
                action = policy(obs_tensor).detach().numpy().squeeze()
                # 把动作转换为目标关节位置
                target_dof_pos = action * action_scale + default_angles

            # 获取物理状态的变化，应用外部扰动，从GUI更新仿真参数
            viewer.sync()

            # 基础的时间同步机制，长时间运行可能与实际时钟产生偏差
            time_until_next_step = m.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
