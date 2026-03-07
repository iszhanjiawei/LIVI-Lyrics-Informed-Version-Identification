#!/bin/bash
# 多GPU多进程并行生成歌词嵌入（chunk 级别 key 格式）
# 配置: GPU 0 运行13个进程，GPU 1 运行13个进程，总计26进程
# save_interval=100，减少临时文件数量

set -e

PROJECT_ROOT="/home/zjw524/projects/LIVI-Lyrics-Informed-Version-Identification"
cd "$PROJECT_ROOT"

# LIVI 虚拟环境的 Python 路径
PYTHON_PATH="/home/zjw524/anaconda3/envs/LIVI/bin/python"

# 总数据量（中文歌曲）
TOTAL_SONGS=43381

# 每个 GPU 的进程数
PROCS_PER_GPU=13
TOTAL_PROCS=$((PROCS_PER_GPU * 2))

# 每个进程处理的歌曲数
SONGS_PER_PROCESS=$((TOTAL_SONGS / TOTAL_PROCS))

echo "=========================================="
echo "启动多GPU多进程歌词嵌入生成（chunk 级别 key）"
echo "=========================================="
echo "总歌曲数: $TOTAL_SONGS"
echo "进程数: $TOTAL_PROCS (GPU0: $PROCS_PER_GPU, GPU1: $PROCS_PER_GPU)"
echo "每进程: ~$SONGS_PER_PROCESS 首歌"
echo "save_interval: 100"
echo "输出格式: chunk 级别 key (songname_chunkindex)"
echo "=========================================="

# 创建日志目录
mkdir -p logs/embedding_generation

# GPU 0 - 13个进程
echo ""
echo "启动 GPU 0 进程..."
for i in $(seq 0 $((PROCS_PER_GPU - 1))); do
    START_IDX=$((i * SONGS_PER_PROCESS))
    END_IDX=$(((i + 1) * SONGS_PER_PROCESS))
    
    LOG_FILE="logs/embedding_generation/gpu0_process${i}.log"
    
    echo "  进程 0-$i: [$START_IDX:$END_IDX] -> $LOG_FILE"
    
    nohup $PYTHON_PATH -u scripts/generate_lyrics_embeddings_parallel.py \
        --gpu-id 0 \
        --start-idx $START_IDX \
        --end-idx $END_IDX \
        > "$LOG_FILE" 2>&1 &
    
    # 错开启动时间，避免同时加载模型导致显存峰值
    sleep 3
done

# GPU 1 - 13个进程
echo ""
echo "启动 GPU 1 进程..."
for i in $(seq 0 $((PROCS_PER_GPU - 1))); do
    GLOBAL_IDX=$((PROCS_PER_GPU + i))
    START_IDX=$((GLOBAL_IDX * SONGS_PER_PROCESS))
    END_IDX=$(((GLOBAL_IDX + 1) * SONGS_PER_PROCESS))
    
    # 最后一个进程处理剩余的所有数据
    if [ $i -eq $((PROCS_PER_GPU - 1)) ]; then
        END_IDX=$TOTAL_SONGS
    fi
    
    LOG_FILE="logs/embedding_generation/gpu1_process${i}.log"
    
    echo "  进程 1-$i: [$START_IDX:$END_IDX] -> $LOG_FILE"
    
    nohup $PYTHON_PATH -u scripts/generate_lyrics_embeddings_parallel.py \
        --gpu-id 1 \
        --start-idx $START_IDX \
        --end-idx $END_IDX \
        > "$LOG_FILE" 2>&1 &
    
    sleep 3
done

echo ""
echo "=========================================="
echo "✅ 所有 $TOTAL_PROCS 个进程已启动！"
echo "=========================================="
echo "查看进程状态: ps aux | grep generate_lyrics_embeddings_parallel"
echo "查看日志: tail -f logs/embedding_generation/gpu*.log"
echo "监控GPU: nvitop"
echo "=========================================="

# 等待所有进程启动完毕后显示状态
sleep 5
echo ""
echo "当前运行的进程:"
ps aux | grep generate_lyrics_embeddings_parallel | grep -v grep | wc -l
echo "个进程正在运行"
