from pathlib import Path

import numpy as np
import torch
from batchgenerators.utilities.file_and_folder_operations import join
from evalutils import SegmentationAlgorithm
from evalutils.validators import (
    UniqueImagesValidator,
    UniquePathIndicesValidator,
)
from nnunetv2.imageio.simpleitk_reader_writer import SimpleITKIO
from nnunetv2.inference.predict_from_raw_data import nnUNetPredictor

from postprocess_probability_maps import postprocess_single_probability_map

RESOURCE_PATH = Path("resources")


class FetalAbdomenSegmentation(SegmentationAlgorithm):
    def __init__(self):
        super().__init__(
            validators=dict(
                input_image=(
                    UniqueImagesValidator(),
                    UniquePathIndicesValidator(),
                )
            ),
        )
        # Path to nnUNet model directory
        self.nnunet_model_dir = RESOURCE_PATH / "nnUNet_results"

        # Initialize the predictor
        self.predictor = self.initialize_predictor()

    def initialize_predictor(self, task="Dataset300_ACOptimalSuboptimal",
                             network="2d", checkpoint="checkpoint_final.pth", folds=(0,)):
        """
        Initializes the nnUNet predictor
        """
        # instantiates the predictor
        predictor = nnUNetPredictor(
            tile_step_size=0.5,
            use_gaussian=True,
            use_mirroring=True,
            device=torch.device('cuda', 0),
            verbose=False,
            verbose_preprocessing=False,
            allow_tqdm=True
        )

        # initializes the network architecture, loads the checkpoint
        predictor.initialize_from_trained_model_folder(
            join(self.nnunet_model_dir,
                 f'{task}/nnUNetTrainer__nnUNetPlans__{network}'),
            use_folds=folds,
            checkpoint_name=checkpoint,
        )
        predictor.dataset_json['file_ending'] = '.mha'

        return predictor

    def predict(self, input_img_path, save_probabilities=True):
        """
        Use trained nnUNet network to generate segmentation masks
        """
        # ideally we would like to use predictor.predict_from_files but this docker container will be called
        # for each individual test case so that this doesn't make sense
        image_np, properties = SimpleITKIO().read_images([input_img_path])
        _, probabilities = self.predictor.predict_single_npy_array(
            image_np, properties, None, None, save_probabilities)

        return probabilities

    def postprocess(self, probability_map):
        """
        Postprocess the nnUNet output to generate the final AC segmentation mask
        """
        # Define the postprocessing configurations
        configs = {
            "soft_threshold": 0.5,
        }

        # Postprocess the probability map
        mask_postprocessed = postprocess_single_probability_map(
            probability_map, configs)
        print('Postprocessing done')
        return mask_postprocessed


def select_fetal_abdomen_mask_and_frame(segmentation_masks: np.ndarray) -> (np.ndarray, int):
    """
    Select the fetal abdomen mask and the corresponding frame number from the segmentation masks
    """
    # Initialize variables to keep track of the largest area and the corresponding 2D image
    largest_area = 0
    selected_image = None

    # Iterate over the 2D images in the 3D array
    for frame in range(len(segmentation_masks)):
        # Calculate the areas for class 1 and class 2 in the current 2D image
        area_class_1 = np.sum(segmentation_masks[frame] == 1)
        area_class_2 = np.sum(segmentation_masks[frame] == 2)

        # If the area of class 1 or class 2 in the current 2D image is larger than the largest area found so far,
        # update the largest area and the selected image
        if area_class_1 > largest_area:
            largest_area = area_class_1
            selected_image = segmentation_masks[frame]
            fetal_abdomen_frame_number = frame
        elif area_class_2 > largest_area:
            largest_area = area_class_2
            selected_image = segmentation_masks[frame]
            fetal_abdomen_frame_number = frame

    # If no 2D image with a positive area was found, provide an empty segmentation mask and set the frame number to -1
    if selected_image is None:
        selected_image = np.zeros_like(segmentation_masks[0])
        fetal_abdomen_frame_number = -1

    # Convert the selected image to a binary mask
    selected_image = (selected_image > 0).astype(np.uint8)
    return selected_image, fetal_abdomen_frame_number
