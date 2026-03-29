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
ENV LD_LIBRARY_PATH=/root/.mujoco/mujoco-3.4.0/lib:/root/.mujoco/mujoco-3.4.0/bin
ENV MUJOCO_GL=egl

SHELL ["/bin/bash", "-c"]
USER root

# -------------------- 安装系统工具 & ROS2 Humble（清华源） --------------------
RUN --mount=type=cache,target=/var/cache/apt \
    set -eux; \
    apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git wget libglib2.0-0 ncurses-term \
    curl software-properties-common ca-certificates gnupg \
    libgl1-mesa-dev libgl1 libglew-dev libosmesa6-dev \
    libyaml-cpp-dev libspdlog-dev libboost-all-dev libglfw3-dev \
    patchelf \
    && add-apt-repository universe \
    && curl -sSL https://mirrors.tuna.tsinghua.edu.cn/rosdistro/ros.key -o /usr/share/keyrings/ros-archive-keyring.gpg \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/ros-archive-keyring.gpg] https://mirrors.tuna.tsinghua.edu.cn/ros2/ubuntu noble main" > /etc/apt/sources.list.d/ros2.list \
    && apt-get update && apt-get install -y --no-install-recommends \
    ros-jazzy-${ROS2_APT_PACKAGE} \
    ros-jazzy-vision-msgs \
    ros-jazzy-rmw-cyclonedds-cpp \
    ros-jazzy-rmw-fastrtps-cpp \
    ros-dev-tools \
    && apt -y autoremove && apt clean autoclean && \
    rm -rf /var/lib/apt/lists/* && \
    echo "source /opt/ros/jazzy/setup.bash" >> ${HOME}/.bashrc

# -------------------- 安装 MuJoCo --------------------
RUN mkdir -p /root/.mujoco && \
    wget https://github.com/google-deepmind/mujoco/releases/download/3.4.0/mujoco-3.4.0-linux-x86_64.tar.gz -O mujoco.tar.gz && \
    tar -zxvf mujoco.tar.gz -C /root/.mujoco && \
    rm mujoco.tar.gz

# -------------------- 复制代码 --------------------
# 先复制 Isaac Lab 基础框架
COPY docker/third_party/IsaacLab/ ${ISAACLAB_PATH}/
# 然后复制你的项目文件（会覆盖 source 和 scripts）
COPY source/himlocolab/ ${ISAACLAB_PATH}/source/himlocolab/
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

# 修复构建工具链：新版 setuptools(72+) 移除了 pkg_resources，导致 flatdict 等旧包无法构建
# 通过 PIP_CONSTRAINT 限制构建隔离环境中的 setuptools 版本，从根本上解决问题
RUN echo "setuptools<72" > /tmp/pip-constraints.txt

RUN --mount=type=cache,target=${DOCKER_USER_HOME}/.cache/pip \
    PIP_CONSTRAINT=/tmp/pip-constraints.txt ${ISAACLAB_PATH}/isaaclab.sh --install

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
    echo 'echo \"==================== IsaacLab 容器已启动 ====================\"' >> ${HOME}/.bashrc && \
    echo 'echo \"容器系统信息：\"' >> ${HOME}/.bashrc && \
    echo 'echo \"   Isaac Sim版本:5.1.0\"' >> ${HOME}/.bashrc && \
    echo 'echo \"   ubuntu版本:24.04.2LTS\"' >> ${HOME}/.bashrc && \
    echo 'echo \"   ROS2版本:Jazzy\"' >> ${HOME}/.bashrc && \
    echo 'echo \"   Python版本:3.11.13\"' >> ${HOME}/.bashrc && \
    echo "shopt -s histappend" >> /root/.bashrc && \
    echo "PROMPT_COMMAND='history -a'" >> /root/.bashrc

WORKDIR ${ISAACLAB_PATH}
