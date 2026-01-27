#!/bin/bash
set -e

cd "$(dirname "$0")"

# 读取容器名
source .env
CONTAINER_NAME="isaac-lab-ros2${DOCKER_NAME_SUFFIX-}"

# 清理模式
if [ "$1" = "-c" ]; then
    echo "清理容器和镜像..."
    docker compose down 2>/dev/null || true
    docker rmi ${CONTAINER_NAME} 2>/dev/null || true
    echo "清理完成！"
    exit 0
fi

# 允许 Docker 访问 X11
xhost +local:docker > /dev/null 2>&1 || echo "警告: 无法设置 xhost，GUI 可能无法显示"

# 检查容器是否运行
if docker ps --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "进入容器: ${CONTAINER_NAME}"
elif docker ps -a --format '{{.Names}}' | grep -q "^${CONTAINER_NAME}$"; then
    echo "启动容器: ${CONTAINER_NAME}"
    docker compose up -d
else
    echo "构建容器: ${CONTAINER_NAME}"
    docker compose up -d
fi

docker exec -it ${CONTAINER_NAME} bash
