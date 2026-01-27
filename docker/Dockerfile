# -------------------- args & env --------------------
ARG ISAACSIM_BASE_IMAGE_ARG
ARG ISAACSIM_VERSION_ARG

FROM ${ISAACSIM_BASE_IMAGE_ARG}:${ISAACSIM_VERSION_ARG}

ARG ROS2_APT_PACKAGE=ros-base

ARG ISAACSIM_ROOT_PATH_ARG
ARG ISAACLAB_PATH_ARG
ARG DOCKER_USER_HOME_ARG

ARG ISAACSIM_VERSION_ARG

ENV ISAACSIM_VERSION=${ISAACSIM_VERSION_ARG}
ENV ISAACSIM_ROOT_PATH=${ISAACSIM_ROOT_PATH_ARG}
ENV ISAACLAB_PATH=${ISAACLAB_PATH_ARG}
ENV DOCKER_USER_HOME=${DOCKER_USER_HOME_ARG}
ENV LANG=C.UTF-8
ENV DEBIAN_FRONTEND=noninteractive
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/root/.mujoco/mujoco210/bin
ENV MUJOCO_GL=egl

SHELL ["/bin/bash", "-c"]
USER root

# -------------------- 安装系统工具 & ROS2 --------------------
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget libglib2.0-0 ncurses-term \
    curl software-properties-common \
    libgl1-mesa-dev \
    libgl1 \
    libglew-dev \
    libosmesa6-dev \
    patchelf \
    && add-apt-repository universe && \
    curl -sSL https://mirrors.tuna.tsinghua.edu.cn/rosdistro/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg && \
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu noble main" | tee /etc/apt/sources.list.d/ros2.list > /dev/null && \
    apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-${ROS2_APT_PACKAGE} \
    ros-jazzy-vision-msgs \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rmw-fastrtps-cpp \
    ros-dev-tools && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/jazzy/setup.bash" >> ${HOME}/.bashrc

# -------------------- 安装 MuJoCo 210 --------------------
RUN mkdir -p /root/.mujoco && \
    wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -zxvf mujoco.tar.gz -C /root/.mujoco && \
    rm mujoco.tar.gz

# -------------------- 复制代码 --------------------
# 先复制 Isaac Lab 基础框架
COPY docker/libs/IsaacLab/ ${ISAACLAB_PATH}/
# 然后复制你的项目文件（会覆盖 source 和 scripts）
COPY source/ ${ISAACLAB_PATH}/source/
COPY scripts/ ${ISAACLAB_PATH}/scripts/
COPY docker/.ros/ ${DOCKER_USER_HOME}/.ros/

# -------------------- 配置 Isaac Lab 环境 --------------------
RUN chmod +x ${ISAACLAB_PATH}/isaaclab.sh && \
    ln -sf ${ISAACSIM_ROOT_PATH} ${ISAACLAB_PATH}/_isaac_sim

RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip install toml

RUN --mount=type=cache,target=/var/cache/apt \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py apt ${ISAACLAB_PATH}/source && \
    ${ISAACLAB_PATH}/isaaclab.sh -p ${ISAACLAB_PATH}/tools/install_deps.py rosdep ${ISAACLAB_PATH}/source && \
    apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    ${ISAACLAB_PATH}/isaaclab.sh --install

# -------------------- 配置系统 GUI 环境 --------------------
RUN apt-get update && apt-get install -y --no-install-recommends \
    libxinerama1 \
    libxcursor1 \
    libxi6 \
    libxrandr2 \
    x11-apps \
    mesa-utils \
    && rm -rf /var/lib/apt/lists/*

# -------------------- 收尾处理 -------------------- 
RUN ${ISAACLAB_PATH}/isaaclab.sh -p -m pip uninstall -y quadprog

RUN echo "export ISAACLAB_PATH=${ISAACLAB_PATH}" >> ${HOME}/.bashrc && \
    echo "alias isaaclab=${ISAACLAB_PATH}/isaaclab.sh" >> ${HOME}/.bashrc && \
    echo "alias python=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias python3=${ISAACLAB_PATH}/_isaac_sim/python.sh" >> ${HOME}/.bashrc && \
    echo "alias pip='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias pip3='${ISAACLAB_PATH}/_isaac_sim/python.sh -m pip'" >> ${HOME}/.bashrc && \
    echo "alias tensorboard='${ISAACLAB_PATH}/_isaac_sim/python.sh ${ISAACLAB_PATH}/_isaac_sim/tensorboard'" >> ${HOME}/.bashrc && \
    echo "export TZ=$(date +%Z)" >> ${HOME}/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /root/.bashrc

WORKDIR ${ISAACLAB_PATH}