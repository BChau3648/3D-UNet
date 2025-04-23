import os 
import numpy as np 
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-p', '--pred', help='Path to predictions')
parser.add_argument('-g', '--gt', help='Path to ground truth')
parser.add_argument('-o', '--output', help='Path to output folder')
args = parser.parse_args()

pred_path = args.pred
gt_path = args.gt
output_path = args.output

