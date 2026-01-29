import time
import mujoco
import mujoco.viewer
import numpy as np

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo

import os

# 项目根目录
HIMLOCO_ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

class Go2Mirror:
    def __init__(self):
        # 关节数据
        self.real_joint_pos = np.zeros(12, dtype=np.float32)
        self.low_state = None
        
        # Go2硬件顺序到训练顺序的映射（反向映射）
        # 硬件: [FR, FL, RR, RL] → 训练: [FL, FR, RL, RR]
        self.hardware_to_training = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
        
    def state_callback(self, msg: LowStateGo):
        """状态回调函数"""
        self.low_state = msg
        for i in range(12):
            self.real_joint_pos[i] = msg.motor_state[i].q
    
    def get_training_order_joints(self):
        training_order = np.zeros(12, dtype=np.float32)
        for training_idx, hardware_idx in enumerate(self.hardware_to_training):
            training_order[training_idx] = self.real_joint_pos[hardware_idx]
        return training_order


if __name__ == "__main__":
    # 配置参数
    net_interface = "enp129s0"
    xml_path = f"{HIMLOCO_ROOT_DIR}/legged_gym/resources/robots/go2/scene.xml"
    
    # 初始化DDS通信
    ChannelFactoryInitialize(0, net_interface)
    
    mirror = Go2Mirror()
    lowstate_subscriber = ChannelSubscriber("rt/lowstate", LowStateGo)
    lowstate_subscriber.Init(mirror.state_callback, 10)
    
    print("连接中...")
    while mirror.low_state is None:
        time.sleep(0.1)
    print("连接成功")
    
    # 加载机器人模型
    m = mujoco.MjModel.from_xml_path(xml_path)
    d = mujoco.MjData(m)
    
    # 实时显示
    with mujoco.viewer.launch_passive(m, d) as viewer:
        while viewer.is_running():
            joint_angles = mirror.get_training_order_joints()
            quat = mirror.low_state.imu_state.quaternion  # [w, x, y, z]
            
            d.qpos[3:7] = quat
            d.qpos[7:] = joint_angles
            
            # 更新仿真
            mujoco.mj_forward(m, d)
            viewer.sync()
            
            # 更新频率
            time.sleep(0.02)
