#!/bin/bash
# 监控 affordance map 生成进度

SAVE_DIR=${1:-"map_pointmamba"}
TOTAL=24544

while true; do
    clear
    echo "╔══════════════════════════════════════════════════════════╗"
    echo "║       Affordance Map 生成进度监控                        ║"
    echo "╠══════════════════════════════════════════════════════════╣"

    # 文件进度
    COUNT=$(ls ${SAVE_DIR}/H3D/pred_contact/ 2>/dev/null | wc -l)
    PERCENT=$((COUNT * 100 / TOTAL))
    BAR_LEN=40
    FILLED=$((PERCENT * BAR_LEN / 100))
    EMPTY=$((BAR_LEN - FILLED))
    BAR=$(printf "%${FILLED}s" | tr ' ' '█')$(printf "%${EMPTY}s" | tr ' ' '░')

    echo "║                                                          ║"
    printf "║  文件进度: %6d / %d  (%3d%%)                      ║\n" $COUNT $TOTAL $PERCENT
    echo "║  [$BAR]  ║"
    echo "║                                                          ║"
    echo "╠══════════════════════════════════════════════════════════╣"

    # GPU0 进度
    if [ -f gen_map_gpu0.log ]; then
        GPU0_BATCH=$(grep "global batch" gen_map_gpu0.log 2>/dev/null | tail -1 | grep -oP '\[\d+/\d+\].*global batch \d+/\d+' | head -1)
        if [ -z "$GPU0_BATCH" ]; then
            GPU0_BATCH="启动中..."
        fi
    else
        GPU0_BATCH="等待启动..."
    fi
    printf "║  GPU 0: %-48s ║\n" "$GPU0_BATCH"

    # GPU1 进度
    if [ -f gen_map_gpu1.log ]; then
        GPU1_BATCH=$(grep "global batch" gen_map_gpu1.log 2>/dev/null | tail -1 | grep -oP '\[\d+/\d+\].*global batch \d+/\d+' | head -1)
        if [ -z "$GPU1_BATCH" ]; then
            GPU1_BATCH="启动中..."
        fi
    else
        GPU1_BATCH="等待启动..."
    fi
    printf "║  GPU 1: %-48s ║\n" "$GPU1_BATCH"

    echo "║                                                          ║"
    echo "╠══════════════════════════════════════════════════════════╣"

    # GPU 显存使用
    echo "║  GPU 显存使用:                                           ║"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null | while read line; do
        IDX=$(echo $line | cut -d',' -f1)
        USED=$(echo $line | cut -d',' -f2 | tr -d ' ')
        TOTAL_MEM=$(echo $line | cut -d',' -f3 | tr -d ' ')
        UTIL=$(echo $line | cut -d',' -f4 | tr -d ' ')
        printf "║    GPU %s: %5s / %5s MiB  (利用率: %3s%%)            ║\n" "$IDX" "$USED" "$TOTAL_MEM" "$UTIL"
    done

    echo "║                                                          ║"
    echo "╚══════════════════════════════════════════════════════════╝"
    echo ""
    echo "按 Ctrl+C 退出监控"

    sleep 3
done
