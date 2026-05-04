#!/bin/bash
# 停止所有训练进程的脚本

echo "正在停止所有训练进程..."

# 1. 强制杀死所有run_multi_gpu_comparison进程
echo "步骤1: 杀死主训练进程..."
pkill -9 -f "run_multi_gpu_comparison.py"
sleep 1

# 2. 检查GPU上是否还有Python进程
echo "步骤2: 检查GPU上的残留进程..."
GPU_PIDS=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null)

if [ -n "$GPU_PIDS" ]; then
    echo "发现GPU上还有进程，正在清理..."
    echo "$GPU_PIDS" | xargs -r kill -9
    sleep 1
else
    echo "GPU上没有残留进程"
fi

# 3. 验证结果
echo ""
echo "验证结果:"
echo "=========================================="

# 检查训练进程
TRAIN_PROCS=$(ps aux | grep -E "python.*run_multi_gpu|python.*train_functions" | grep -v grep | wc -l)
echo "训练进程数量: $TRAIN_PROCS"

# 检查GPU使用情况
echo ""
echo "GPU状态:"
nvidia-smi --query-gpu=index,utilization.gpu,memory.used --format=csv,noheader

# 检查GPU上的进程
GPU_PROC_COUNT=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
echo ""
echo "GPU上的进程数量: $GPU_PROC_COUNT"

echo "=========================================="

if [ $TRAIN_PROCS -eq 0 ] && [ $GPU_PROC_COUNT -eq 0 ]; then
    echo "✅ 所有训练进程已成功停止！"
else
    echo "⚠️  可能还有残留进程，请手动检查"
fi
