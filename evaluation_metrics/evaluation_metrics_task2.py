import numpy as np
import pandas as pd
import nibabel as nib
import pathlib as plb
import csv
import sys
import os
import glob
import argparse
from scipy import integrate


parser = argparse.ArgumentParser()
parser.add_argument("-g", "--gt_path", required=True, help="Path to the nifti GT labels folder")
parser.add_argument("-p", "--prediction_path", required=False, help="Path to the nifti Pred labels folder")
args = parser.parse_args()

def nii2numpy(nii_path):
    # input: path of NIfTI segmentation file, output: corresponding numpy array and voxel_vol in ml
    mask_nii = nib.load(str(nii_path))
    mask = mask_nii.get_fdata()
    pixdim = mask_nii.header['pixdim']   
    voxel_vol = pixdim[1]*pixdim[2]*pixdim[3]/1000
    return mask, voxel_vol

def false_pos_pix(ground_truth: np.ndarray, prediction: np.ndarray):
    """
    Count the number of false positive pixels for a single label.
    Returns zero if the prediction array is empty.
    Args:
        prediction (np.ndarray): The predicted array.
        ground_truth (np.ndarray): The ground truth array.
    returns:
        float: The number of false positive pixels which do not overlap with the ground truth.

    """
    
    if prediction.sum() == 0:
        return 0

    false_positive = ((prediction == 1) & (ground_truth == 0)).sum()
    return float(false_positive)

def false_neg_pix(ground_truth: np.ndarray, prediction: np.ndarray):
    """
    Count the number of false negative pixels for a single label.
    Returns nan if the ground truth array is empty.
    Args:
        prediction (np.ndarray): The predicted binary mask for the label.
        ground_truth (np.ndarray): The ground truth binary mask for the label.
    Returns:
        float: The number of false negative pixels, which do not overlap with the prediction.

    """
    
    if ground_truth.sum() == 0:
        return np.nan

    false_negative = ((ground_truth == 1) & (prediction == 0)).sum()
    return float(false_negative)

def dice_score(ground_truth: np.ndarray, prediction: np.ndarray) -> float:
    """
    Calculate the Dice score for a single label between the prediction and ground truth arrays.
    Returns NaN if the ground truth array is empty.

    Args:
        prediction (np.ndarray): The predicted binary mask for the label.
        ground_truth (np.ndarray): The ground truth binary mask for the label.

    Returns:
        float: The Dice score for the label between the prediction and ground truth arrays.
    """
    if ground_truth.sum() == 0:
        return np.nan

    intersection = (ground_truth * prediction).sum()
    union = ground_truth.sum() + prediction.sum()
    dice_score = 2 * intersection / union

    return dice_score


def compute_metrics(nii_gt_path, nii_pred_path):
    """
    
    For each label in the ground truth:
      - Compute Dice
      - Compute false positive volume
      - Compute false negative volume
    Returns:
        dices: list of  Dice scores for all lesion overlaps
        fp_vols: list of false positive volumes
        fn_vols: list of total false negative volumes
    """

    gt_array, voxel_vol = nii2numpy(nii_gt_path)
    pred_array, voxel_vol = nii2numpy(nii_pred_path)

    
    num_classes = np.unique(gt_array)
    dice_scores = []
    fp_vols = []
    fn_vols = []
    pred_labels_used = []
    
    for cls in num_classes:
        if cls == 0:
            continue
        gt_cls = (gt_array == cls).astype(np.uint8)

        # Find which prediction labels overlap with this GT lesion
        overlapped_pred_labels = np.unique(pred_array[gt_cls == 1])
        overlapped_pred_labels = overlapped_pred_labels[overlapped_pred_labels != 0]

        if len(overlapped_pred_labels) == 0:
            # FN: this GT lesion wasn't predicted at all
            dice_scores.append(0.0)
            fp_vols.append(0.0)
            fn_vols.append(gt_cls.sum() * voxel_vol)
            continue

        # One or more predicted lesions overlap GT lesion
        for pred_lab in overlapped_pred_labels:
            pred_cls = (pred_array == pred_lab).astype(np.uint8)
            dice = dice_score(gt_cls, pred_cls)
            fp = false_pos_pix(gt_cls, pred_cls) * voxel_vol
            fn = false_neg_pix(gt_cls, pred_cls) * voxel_vol
            dice_scores.append(dice)
            fp_vols.append(float(fp))
            fn_vols.append(float(fn))
            pred_labels_used.append(pred_lab)

    # FPs: predicted lesions that never overlapped with any GT lesion
    pred_labels = np.unique(pred_array)
    pred_labels = pred_labels[pred_labels != 0]
    pred_labels_remaining = set(pred_labels) - set(pred_labels_used)

    for pred_label in sorted(pred_labels_remaining):
        pred_cls = (pred_array == pred_label).astype(np.uint8)
        overlap = np.sum(gt_array[pred_cls == 1] > 0)
        if overlap == 0:
            if gt_array.sum() == 0:
                dice_scores.append(np.nan)
                fn_vols.append(np.nan)
            else:
                dice_scores.append(0.0)
                fn_vols.append(0.0)
            fp_vols.append(pred_cls.sum() * voxel_vol)
            

    return np.nanmean(dice_scores), np.nanmean(fp_vols), np.nanmean(fn_vols)


def get_metrics(gt_folder_path, pred_folder_path):
    # Get metrics for a single test case and aggregate.
    # input: path of NIfTI segmentation files (ground truth and prediction)
    # output: Dice score, false positive volume, false negative volume
    csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
    with open("metrics.csv", "w", newline='') as f:
        writer = csv.writer(f, delimiter=',')
        writer.writerow(csv_header)
        
        # loop over all test cases
        for nii_gt_path in glob.glob(os.path.join(gt_folder_path, '*.nii.gz')): 
            nii_pred_path = os.path.join(pred_folder_path, os.path.basename(nii_gt_path))
            if not os.path.exists(nii_pred_path):
                print(f"Missing prediction for: {os.path.basename(nii_gt_path)}")
                continue
            nii_gt_path = plb.Path(nii_gt_path)
            nii_pred_path = plb.Path(nii_pred_path)
            dice_sc, false_pos_vol, false_neg_vol = compute_metrics(nii_gt_path, nii_pred_path)
            writer.writerow([
                os.path.basename(nii_gt_path),
                dice_sc,
                false_pos_vol,
                false_neg_vol
            ])


if __name__ == "__main__":
    # example usage to call evaluation on complete test cohort and a single test case
    # calls and interfaces are kept similar to grand-challenge.org execution environment

    # assumed directory structure (nii_gt files and nii_pred files can also be stored in the same folder):
    # /path/to/gt/labels
    #   |-- test_case_1
    #       |-- nii_gt_path_file_00.nii.gz
    #   |-- test_case_2
    #       |-- nii_gt_path_file_00.nii.gz
    #       |-- nii_gt_path_file_01.nii.gz
    #   |-- ...
    #
    # /path/to/pred/labels
    #   |-- test_case_1
    #       |-- nii_gt_path_file_pred_00.nii.gz
    #   |-- test_case_2
    #       |-- nii_gt_path_file_pred_00.nii.gz
    #       |-- nii_gt_path_file_pred_01.nii.gz
    #       |-- ...
    #   |-- ...
    
    # complete test cohort: 
    #  python evaluation_metrics_task2.py -c -gt_path /path/to/gt/labels -prediction_path /path/to/pred/labels
    # single test case:     
    #  python evaluation_metrics_task2.py -gt_path /path/to/gt/labels/test_case_1 -prediction_path /path/to/pred/labels/test_case_1
    

    gt_folder_path = args.gt_path
    pred_folder_path = args.prediction_path
    

    if os.path.isdir(gt_folder_path) and os.path.isdir(pred_folder_path):
        if args.c:
            # complete test cohort

            # prepare complete test cohort csv file
            csv_header = ['gt_name', 'dice_sc', 'false_pos_vol', 'false_neg_vol']
            df_cohort = pd.DataFrame(columns=csv_header)

            # loop over all test cases
            for test_case in glob.glob(gt_folder_path):
                gt_folder = os.path.join(gt_folder_path, test_case)
                pred_folder = os.path.join(pred_folder_path, test_case)

                # compute metrics
                get_metrics(gt_folder, pred_folder)
                # aggregate metrics
                df = pd.read_csv('metrics.csv')
                df['gt_name'] = test_case
                df_cohort = pd.concat([df_cohort, df], ignore_index=True)

            # save cohort metrics
            df_cohort.to_csv('metrics_cohort.csv', index=False)
            # remove temporary file
            os.remove('metrics.csv')
            # aggregate metrics
            numeric_cols = ['dice_sc', 'false_pos_vol', 'false_neg_vol']
            metrics = df_cohort[numeric_cols]
            print(metrics.mean())
        
        else:
            # single test case
            get_metrics(gt_folder_path, pred_folder_path)

            # aggregate metrics
            df = pd.read_csv('metrics.csv')
            numeric_cols = ['dice_sc', 'false_pos_vol', 'false_neg_vol']
            metrics = df[numeric_cols]
            print(metrics.mean())
            