import os
import shutil
import numpy as np
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--train_path', help='Path to training folder (containing data)')
parser.add_argument('--val_path', help='Path to validation folder')
args = parser.parse_args()

train_path = args.train_path
val_path = args.val_path

cases = [f for f in os.listdir(train_path) if f.endswith('.h5')]
cases = np.array(sorted(cases))
idx = np.arange(len(cases)).astype(int)
np.random.seed(0)
np.random.shuffle(idx)
val_idx = idx[:int(len(cases) * 0.2)]

counter = 0
for i,case in enumerate(cases[val_idx]):
    og_path = os.path.join(train_path, case)
    new_path = os.path.join(val_path, case)
    shutil.move(og_path, new_path)
    counter += 1
print(f"Moved {counter}/{len(cases)} ({counter/len(cases)}%) validation files")