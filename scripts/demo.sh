#!/bin/bash

export online=false
export carla=false
export tensorfi=false
export tensorfisingleinjection=false
export DATASET_DIR="./data/datasets/driving_dataset"
export MODEL_FILE="./data/models/nvidia/model.ckpt"

while test $# -gt 0; do
  case "$1" in
    -h|--help)
      echo "Usage: ./scripts/demo.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help           show brief help"
      echo "-model_file          model files for restoring PilotNet, default './data/models/nvidia/model.ckpt'"
      echo "-online              run the demo on a live webcam feed, default demo on dataset"
      echo "-dataset_dir         dataset given input images of the road ahead, default './data/datasets/driving_dataset'"
      exit 0
      ;;
    -model_file)
      export MODEL_FILE="$2"
      shift
      shift
      ;;
    -carla)
      export carla=true
      shift
      ;;
    -tensorfi)
      export tensorfi=true
      shift
      ;;
    -tensorfisingle)
      export tensorfisingleinjection=true
      shift
      ;;
    -online)
      export online=true
      shift
      ;;
    -dataset_dir)
      export DATASET_DIR="$2"
      shift
      shift
      ;;
    *)
      echo "Usage: ./scripts/demo.sh [options]"
      echo " "
      echo "options:"
      echo "-h, --help           show brief help"
      echo "-model_file          model files for restoring PilotNet, default './data/models/nvidia/model.ckpt'"
      echo "-online              run the demo on a live webcam feed, default demo on dataset"
      echo "-dataset_dir         dataset given input images of the road ahead, default './data/datasets/driving_dataset'"
      exit 0
      ;;
  esac
done

if [ $online == true ]; then
  python ./src/run_capture.py \
    --model $MODEL_FILE
elif [ $carla == true ]; then
  python ./src/run_carla.py \
    --model $MODEL_FILE
elif [ $tensorfisingleinjection == true ]; then
  python ./src/run_tensorfi_model_injection.py \
    --model_file $MODEL_FILE
elif [ $tensorfi == true ]; then
  python ./src/run_tensorfi.py \
    --model_file $MODEL_FILE
else
  python ./src/run_dataset.py \
    --model_file $MODEL_FILE \
    --dataset_dir $DATASET_DIR
fi
