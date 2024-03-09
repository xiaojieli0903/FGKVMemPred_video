#!/usr/bin/env bash
CONFIG=$1
PORT=$2
GPUS=$3
LIST_PATH=$4
PATH_PREFIX=$5

export NCCL_P2P_LEVEL=NVL

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python -u tools/run_net.py --cfg $CONFIG \
--init_method tcp://localhost:$PORT \
--opts DATA.PATH_TO_DATA_DIR $LIST_PATH \
BN.NUM_SYNC_DEVICES $GPUS \
NUM_GPUS $GPUS \
DATA.PATH_PREFIX $PATH_PREFIX \
DATA_LOADER.NUM_WORKERS 8