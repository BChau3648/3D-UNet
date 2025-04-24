import os
from tqdm import tqdm
import numpy as np
import torch
import nibabel as nib
from nibabel.processing import resample_to_output
from nibabel.orientations import io_orientation, ornt_transform, apply_orientation
import h5py
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-i', '--images_path', help='Path to images')
parser.add_argument('-l', '--labels_path', help='Path to labels')
parser.add_argument('-s', '--stats_path', help='Path to preprocessing statistics')
parser.add_argument('-o', '--output_path', help='Path to output folder')
args = parser.parse_args()

images_path = args.images_path
labels_path = args.labels_path
stats_path = args.stats_path
output_path = args.output_path

# Getting sorted image and label files
image_files = [os.path.join(images_path, f) for f in os.listdir(images_path) if f.endswith('.nii.gz')]
label_files = [os.path.join(labels_path, f) for f in os.listdir(labels_path) if f.endswith('.nii.gz')]
image_files.sort()
label_files.sort()

n = len(image_files)

# Loading preprocessing stats
preprocess_stats = np.load(stats_path)
pixdims = preprocess_stats['pixdims']
target_intensities = preprocess_stats['target_intensities']
shapes = preprocess_stats['shapes']
# Calculating clipping range, mean and std, and target voxel spacing
minmax_intensities = [np.percentile(target_intensities, 0.5), np.percentile(target_intensities, 99.5)]
mean_intensities = np.mean(target_intensities)
std_intensities = np.std(target_intensities)
# Note: The only difference from the regular preprocess is that the target spacing is set to [1,1,1]
target_spacing = np.array([1., 1., 1.])

# Saving preprocessed image shapes to determine patch size
preprocessed_shapes = np.zeros((len(image_files), 3))
# Iterating over all of the data
for i in tqdm(range(n)):
    # resampling, clipping, and normalizing image
    img = nib.load(image_files[i])
    resamp_img = resample_to_output(img, target_spacing)
    resamp_img_arr = resamp_img.get_fdata()
    clipped_img_arr = np.clip(resamp_img_arr, minmax_intensities[0], minmax_intensities[1])
    norm_img_arr = (clipped_img_arr - mean_intensities) / std_intensities
    
    # resampling label and making sure labels are the same
    label = nib.load(label_files[i])
    resamp_label = resample_to_output(label, target_spacing, order=0)
    resamp_label_arr = resamp_label.get_fdata()
    # Note: below is for two classes -- one foreground and one background
    assert len(np.unique(resamp_label_arr)) == 2
    
    # reorient resampled image and label
    curr_orientation = io_orientation(resamp_img.affine)
    target_orientation = io_orientation(img.affine)
    transform = ornt_transform(curr_orientation, target_orientation)
    aligned_img_arr = apply_orientation(norm_img_arr, transform)
    aligned_label_arr = apply_orientation(resamp_label_arr, transform)
    # flipping preprocessed scan to align with ZYX
    preprocess_img_arr = np.swapaxes(aligned_img_arr, 0, 2)
    preprocess_label_arr = np.swapaxes(aligned_label_arr, 0, 2)
    
    # one hot encoding labels
    onehot_label_arr = torch.from_numpy(preprocess_label_arr.copy()).long()
    onehot_label_arr = torch.zeros(2, *onehot_label_arr.shape).scatter_(0, onehot_label_arr.unsqueeze(0), 1).numpy()
    
    # saving preprocessed image shapes 
    preprocessed_shapes[i,:] = preprocess_img_arr.shape
    
    # saving arrays to h5 files
    filename = image_files[i].split('/')[-1].split('.')[0] + '.h5'
    with h5py.File(os.path.join(output_path, filename), 'w') as hf:
        hf.create_dataset('raw', data=preprocess_img_arr, compression="gzip")
        hf.create_dataset('label', data=onehot_label_arr, compression="gzip")

np.savez(os.path.join(output_path, 'preprocessed_shapes.npz'), preprocessed_shapes)