#!/usr/bin/env python3
"""
Medical Image Defacing Tool for CT Scans

This script performs automatic defacing of CT images by removing facial features
and identifiable anatomical structures while preserving medical data integrity.

Prerequisites:
CT segmentation masks for brain, skull, body, and clavicles must be available (e.g. using totalsegmentator, https://github.com/wasserth/TotalSegmentator).

"""

import os
import nibabel as nib
import numpy as np
import glob
import logging
import argparse
from pathlib import Path
from typing import List, Tuple, Optional
import sys
from scipy.ndimage import label, gaussian_filter

def keep_largest_connected_component(mask: np.ndarray) -> np.ndarray:
    """
    Keep only the largest connected component in a binary mask.
    
    Args:
        mask (np.ndarray): Binary mask array
        
    Returns:
        np.ndarray: Binary mask containing only the largest connected component
    """
    # Label connected components
    labeled_mask, num_features = label(mask)
    
    if num_features == 0:
        return mask.astype(bool)
    
    # Count the size of each component
    component_sizes = np.bincount(labeled_mask.ravel())
    
    # Ignore the background (component 0)
    component_sizes[0] = 0
    
    # Find the largest component
    largest_component = component_sizes.argmax()
    
    # Create a mask for the largest component
    largest_component_mask = (labeled_mask == largest_component)
    
    return largest_component_mask

def deface(file_path: str, seg_path: str, body_seg_dir: str, output_dir: str) -> None: 
    """
    Remove facial features from CT images based on anatomical segmentation masks.
    
    This function performs defacing by:
    1. Loading anatomical segmentation masks (brain, skull, body, clavicles)
    2. Identifying facial regions anterior to brain/skull structures
    3. Applying Gaussian blur to facial regions while preserving medical data
    4. Removing anterior artifacts and adjusting HU values
    
    Args:
        file_path (str): Path to the input NIfTI CT file
        seg_path (str): Path to directory containing anatomical segmentation masks
        body_seg_dir (str): Path to directory containing body segmentation mask
        output_dir (str): Directory to save the defaced NIfTI file
        
    Raises:
        FileNotFoundError: If required segmentation masks are missing
        ValueError: If image dimensions don't match between CT and masks
    """
    try:
        # Load the input NIfTI file
        img = nib.load(file_path)
        original_affine = img.affine
        data = img.get_fdata()
        
        # Validate data
        if data.ndim != 3:
            raise ValueError(f"Expected 3D image, got {data.ndim}D")

        # Load anatomical segmentation masks
        required_masks = ['brain.nii.gz', 'skull.nii.gz']
        for mask_name in required_masks:
            mask_path = os.path.join(seg_path, mask_name)
            if not os.path.exists(mask_path):
                raise FileNotFoundError(f"Required mask not found: {mask_path}")

        brain_mask = nib.load(os.path.join(seg_path, 'brain.nii.gz'))
        brain_mask_data = brain_mask.get_fdata()

        skull_mask = nib.load(os.path.join(seg_path, 'skull.nii.gz'))
        skull_mask_data = skull_mask.get_fdata()

        body_mask = nib.load(os.path.join(body_seg_dir, 'body.nii.gz'))
        body_mask_data = body_mask.get_fdata()
        body_mask_data = keep_largest_connected_component(body_mask_data)

        # Load clavicle masks (optional - may not be present in all cases)
        clavicula_left_path = os.path.join(seg_path, 'clavicula_left.nii.gz')
        clavicula_right_path = os.path.join(seg_path, 'clavicula_right.nii.gz')
        
        if os.path.exists(clavicula_left_path):
            clavicula_left_mask = nib.load(clavicula_left_path)
            clavicula_left_mask_data = clavicula_left_mask.get_fdata()
            clavicula_left_mask_data = keep_largest_connected_component(clavicula_left_mask_data)
        else:
            clavicula_left_mask_data = np.zeros_like(data)
            
        if os.path.exists(clavicula_right_path):
            clavicula_right_mask = nib.load(clavicula_right_path)
            clavicula_right_mask_data = clavicula_right_mask.get_fdata()
            clavicula_right_mask_data = keep_largest_connected_component(clavicula_right_mask_data)
        else:
            clavicula_right_mask_data = np.zeros_like(data)

        # Combine clavicle masks
        clavicula_mask = clavicula_left_mask_data + clavicula_right_mask_data
        
        # Find anatomical landmarks for defacing boundaries
        # Highest slice containing clavicle (superior boundary)
        if np.sum(clavicula_mask) == 0:
            highest_clavicula_slice = 0  # No clavicle found, use first slice
        else:
            highest_clavicula_slice = np.where(clavicula_mask.sum(axis=(0, 1)) > 0)[0][-1]

        # Most anterior slice containing clavicle (anterior boundary)  
        if np.sum(clavicula_mask) == 0:
            anterior_clavicula_slice = 0  # No clavicle found, use first slice
        else:
            anterior_clavicula_slice = np.where(clavicula_mask.sum(axis=(0, 2)) > 0)[0][0]

        # Find extremal points of clavicles for lateral boundaries
        if np.sum(clavicula_right_mask_data) == 0:
            inner_point_right_clavicula = 0
            outer_point_right_clavicula = 0
        else:
            inner_point_right_clavicula = np.where(clavicula_right_mask_data.sum(axis=(1, 2)) > 0)[0][-1]
            outer_point_right_clavicula = data.shape[0] - 1 - np.where(clavicula_right_mask_data.sum(axis=(1, 2))[::-1] > 0)[0][-1]

        if np.sum(clavicula_left_mask_data) == 0:
            inner_point_left_clavicula = data.shape[0] - 1 
            outer_point_left_clavicula = data.shape[0] - 1
        else:
            inner_point_left_clavicula = data.shape[0] - 1 - np.where(clavicula_left_mask_data.sum(axis=(1, 2))[::-1] > 0)[0][-1]   
            outer_point_left_clavicula = np.where(clavicula_left_mask_data.sum(axis=(1, 2)) > 0)[0][-1]
        
        mid_point_right_clavicula = (inner_point_right_clavicula + outer_point_right_clavicula) // 2
        mid_point_left_clavicula = (inner_point_left_clavicula + outer_point_left_clavicula) // 2

        # Create defacing mask by identifying facial regions
        brain_skull_mask = brain_mask_data + skull_mask_data
        
        # Initialize output mask for facial regions
        out_mask = np.zeros_like(brain_skull_mask)
        A, H, W = out_mask.shape  

        # Iterate over each axial slice to identify facial regions
        for x in range(W): 
            if x > highest_clavicula_slice and brain_mask_data[:, :, x].sum() != 0:
                for y in range(A): 
                    # Find anterior body boundary with safety margin
                    body_anterior_idx = np.argmax(body_mask_data[y, :, x] >= 1) + 40
                    brain_skull_row = brain_skull_mask[y, :, x]  
                    brain_row = brain_mask_data[y, :, x]
                    
                    # Check if there's any brain/skull in the row
                    if np.any(brain_skull_row):
                        # Find the most anterior brain voxel
                        if np.any(brain_row >= 1):
                            most_anterior_idx = min(np.argmax(brain_row >= 1), body_anterior_idx)
                        else:
                            most_anterior_idx = body_anterior_idx
                        
                        # Mark facial region anterior to brain/skull
                        out_mask[y, :most_anterior_idx, x] = 1

        # Apply lateral boundaries based on clavicle positions
        out_mask[:mid_point_right_clavicula, :, :] = 0
        out_mask[mid_point_left_clavicula:, :, :] = 0

        # Extend defacing to lower slices to ensure complete facial removal
        out_mask_lower = np.copy(out_mask)
        nonzero_slices = np.where(out_mask.sum(axis=(0, 1)) > 0)[0]
        
        if nonzero_slices.size > 0:
            lowest_slice = nonzero_slices[0]    
            upper_limit = min(lowest_slice + 40, out_mask.shape[2])
            
            if lowest_slice > 0:
                extension_slices = upper_limit - lowest_slice
                start_idx = max(0, lowest_slice - extension_slices)
                out_mask_lower[:, :, start_idx:lowest_slice] = np.logical_or.reduce(
                    out_mask[:, :, lowest_slice:upper_limit], axis=2, keepdims=True
                )
        
        out_mask = out_mask + out_mask_lower

        # Apply final anatomical boundaries
        out_mask[:, :, :highest_clavicula_slice] = 0
        out_mask[:, anterior_clavicula_slice:, :] = 0

        logging.info(f"Anterior clavicle slice: {anterior_clavicula_slice}")

        # Apply defacing: blur facial regions while preserving medical data
        out_mask = out_mask.astype(bool)
        
        # Parameters for defacing
        GAUSSIAN_SIGMA = 20  # Blur strength for facial features
        HU_OFFSET = 1024     # Standard CT HU offset correction
        
        filtered_data = gaussian_filter(data, sigma=GAUSSIAN_SIGMA)
        data[out_mask] = filtered_data[out_mask]

        # Remove anterior artifacts
        data[:, :20, highest_clavicula_slice:] = 0
                    
        # Apply HU value correction if necessary
        data = data - HU_OFFSET

        # Save the defaced image
        output_file_path = os.path.join(output_dir, os.path.basename(file_path))
        new_img = nib.Nifti1Image(data, original_affine, img.header)
        
        logging.info(f"Saving defaced file to: {output_file_path}")
        nib.save(new_img, output_file_path)
        
    except FileNotFoundError as e:
        logging.error(f"File not found: {e}")
        raise
    except ValueError as e:
        logging.error(f"Invalid data: {e}")
        raise
    except Exception as e:
        logging.error(f"Failed to process {file_path}: {e}")
        raise

def process_files(input_dir: str, seg_dir: str, body_seg_dir: str, output_dir: str) -> None:
    """
    Process all NIfTI files in the input directory and perform defacing.
    
    Args:
        input_dir (str): Directory containing the input NIfTI files
        seg_dir (str): Directory containing the anatomical segmentation masks
        body_seg_dir (str): Directory containing the body segmentation masks
        output_dir (str): Directory to save the defaced NIfTI files
        
    Raises:
        ValueError: If input directories don't exist
        RuntimeError: If no valid files found for processing
    """
    # Validate input directories
    for directory, name in [(input_dir, "input"), (seg_dir, "segmentation"), (body_seg_dir, "body segmentation")]:
        if not os.path.exists(directory):
            raise ValueError(f"{name.capitalize()} directory does not exist: {directory}")
    
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Get a list of all NIfTI files in the input directory
    file_paths = glob.glob(os.path.join(input_dir, '*.nii.gz'))
    
    if not file_paths:
        raise RuntimeError(f"No NIfTI files found in input directory: {input_dir}")
    
    logging.info(f"Found {len(file_paths)} files to process")

    # Process each file
    processed_count = 0
    failed_count = 0
    
    for file_path in file_paths:
        try:
            logging.info(f"Processing: {os.path.basename(file_path)}")
            
            # Get the corresponding segmentation paths
            base_name = os.path.basename(file_path).replace('.nii.gz', '')
            seg_path = os.path.join(seg_dir, base_name)
            body_seg_path = os.path.join(body_seg_dir, base_name)
            
            if not os.path.exists(seg_path):
                logging.warning(f"Segmentation path does not exist for {file_path}: {seg_path}")
                failed_count += 1
                continue
                
            if not os.path.exists(body_seg_path):
                logging.warning(f"Body segmentation path does not exist for {file_path}: {body_seg_path}")
                failed_count += 1
                continue

            # Perform defacing
            deface(file_path, seg_path, body_seg_path, output_dir)
            processed_count += 1
            
        except Exception as e:
            logging.error(f"Failed to process {file_path}: {e}")
            failed_count += 1
            continue
    
    logging.info(f"Processing complete. Successfully processed: {processed_count}, Failed: {failed_count}")


def main():
    """Main function with argument parsing for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Medical Image Defacing Tool for CT Scans",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python defacing.py --input /path/to/ct_images --seg /path/to/segmentations --body /path/to/body_segs --output /path/to/output
    
    python defacing.py --input input_dir --seg seg_dir --body body_dir --output output_dir --verbose
        """
    )
    
    parser.add_argument('--input', '-i', required=True,
                       help='Directory containing input NIfTI CT files')
    parser.add_argument('--seg', '-s', required=True,
                       help='Directory containing anatomical segmentation masks')
    parser.add_argument('--body', '-b', required=True,
                       help='Directory containing body segmentation masks')
    parser.add_argument('--output', '-o', required=True,
                       help='Output directory for defaced images')
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    try:
        process_files(args.input, args.seg, args.body, args.output)
        logging.info("Defacing completed successfully!")
    except Exception as e:
        logging.error(f"Defacing failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
else:
    # For backwards compatibility when script is imported
    # Configure basic logging for import usage
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


    # If running as imported module, you can still use e.g. hardcoded paths below
    # process_files(
    #     input_dir='/path/to/input_dir',
    #     seg_dir='/path/to/seg_dir',
    #     body_seg_dir='/path/to/body_seg_dir',
    #     output_dir='/path/to/output_dir'
    # )
