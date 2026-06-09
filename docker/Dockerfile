# docker/Dockerfile
# Base: vLLM 官方 ROCm 镜像，ROCm 7.0
FROM rocm/vllm-rocm:v0.18.1

WORKDIR /workspace

# 装自定义层
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 预热 vllm 编译缓存（首次 import 触发）
RUN python -c "import vllm; print(vllm.__version__)"
