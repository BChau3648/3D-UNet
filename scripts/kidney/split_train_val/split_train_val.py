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

cases = np.array([f for f in os.listdir(train_path) if f.endswith('.h5')])
cases = sorted(cases)
idx = np.arange(len(cases))
np.random.seed(0)
np.random.shuffle(idx)
test_idx = idx[:int(len(cases) * 0.2)]

counter = 0
for i,case in enumerate(cases[test_idx]):
    test_image = images_path + '/' + case[:-7] + '_0000' + case[-7:]
    new_test_image = test_images_path + '/' + case[:-7] + '_0000' + case[-7:]
    test_label = labels_path + '/' + case
    new_test_label = test_labels_path + '/' + case
    shutil.copy2(test_image, new_test_image)
    shutil.copy2(test_label, new_test_label)
    counter+=1 

    shutil.move(test_image, new_test_image)
    shutil.move(test_label, new_test_label)
    counter += 1
print(f"Moved {counter}/{len(cases)} ({counter/len(cases)}%) testing sets")