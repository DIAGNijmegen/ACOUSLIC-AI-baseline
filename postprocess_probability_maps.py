
import numpy as np
from nnunetv2.postprocessing.remove_connected_components import (
    remove_all_but_largest_component_from_segmentation,
)


def get_binary_masks(softmax_array, thres, class_label):
    # get all binary segmentation maps - threshold operation
    binary_masks = softmax_array[class_label].copy()

    binary_masks[binary_masks >= thres] = 1
    binary_masks[binary_masks < thres] = 0

    return binary_masks.astype(np.uint8)


def get_positive_frames(mask):
    frames_positive = []
    for fr in range(len(mask)):
        if np.any(mask[fr] != 0):
            frames_positive.append(fr)
    return frames_positive


def merge_annotations(existing_labels, new_labels, priority_label=None):
    # Check if the arrays are 2D (single frame) or 3D (multiple frames)
    if len(existing_labels.shape) == 2:
        # Convert to 3D for consistent handling
        existing_labels = existing_labels[np.newaxis, ...]
        new_labels = new_labels[np.newaxis, ...]

    # Create a mask for the overlapping regions
    overlap_mask = (existing_labels != 0) & (new_labels != 0)

    # Initialize the merged labels with the existing labels
    merged_labels = existing_labels.copy()
    # Handle the non-overlapping regions
    print('Merging non-overlapping regions')
    merged_labels[new_labels != 0] = new_labels[new_labels != 0]

    if np.any(overlap_mask):
        print('Found overlapping labels. Merging...')
        # Handle the overlapping regions
        if priority_label is not None:
            # If a priority label is specified, use it for the overlapping regions
            merged_labels[overlap_mask] = priority_label
        else:
            # If no priority label is specified, choose the label with more pixels
            existing_pixels = np.sum(
                existing_labels == existing_labels[overlap_mask])
            new_pixels = np.sum(new_labels == new_labels[overlap_mask])
            merged_labels[overlap_mask] = np.where(
                existing_pixels >= new_pixels, existing_labels[overlap_mask], new_labels[overlap_mask])
    # If the input was 2D, return the 2D result
    if len(existing_labels.shape) == 2:
        return merged_labels[0, ...]
    return merged_labels


def postprocess_single_probability_map(softmax_prob_map, configs):
    # Define fetal structures labels
    labels_dict = dict(optimal=1, suboptimal=2)

    # Copy the input probability map
    softmax_maps = softmax_prob_map.copy()
    # Apply threshold
    softmax_maps[softmax_maps < configs["soft_threshold"]] = 0

    # Find the class with the maximum probability at each pixel across all channels
    # This will have shape [n_frames, H, W]
    masks = np.argmax(softmax_maps, axis=0)
    masks = masks.astype(np.uint8)

    # keep the largest connected component for each class
    masks_postprocessed = remove_all_but_largest_component_from_segmentation(
        masks, labels_or_regions=labels_dict.values())
    return masks_postprocessed
