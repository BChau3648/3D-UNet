#!/bin/bash
export MPLCONFIGDIR=/tmp/matplotlib
export CUDA_VISIBLE_DEVICES=0,1

train3dunet --config $1