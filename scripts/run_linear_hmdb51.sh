#!/usr/bin/env bash
CONFIG=$1
CHECKPOINT=$2
LIST_PATH=$3
PATH_PREFIX=$4
PORT=$2

export NCCL_P2P_LEVEL=NVL

python -u tools/run_net.py \
--cfg $CONFIG \
--init_method tcp://localhost:$PORT \
--opts TRAIN.CHECKPOINT_FILE_PATH $CHECKPOINT \
OUTPUT_DIR $2_hmdb51_linear \
DATA.PATH_TO_DATA_DIR $LIST_PATH \
DATA.PATH_PREFIX $PATH_PREFIX
MODEL.NUM_CLASSES 51