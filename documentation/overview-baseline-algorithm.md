# An overview of the baseline algorithm
The ACOUSLIC-AI baseline algorithm is a deep learning-based classification and segmentation model based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet/). It was trained for one fold with an 80/20 train/validation split on the ACOUSLIC-AI [Training and Development Dataset](https://doi.org/10.5281/zenodo.11005384). The code to run inference using this baseline method is in `inference.py`.

## Input and output interfaces
The algorithm loads and processes a stack of 2D B-mode ultrasound images, identifies the optimal frame (image) for fetal abdomen circumference measurement and provides a single 2D binary fetal abdomen segmentation mask for the selected frame.

The algorithm reads the following input:
- Stacked fetal ultrasound images at `"/input/images/stacked-fetal-ultrasound/<uuid>.mha"`: Stacked 2D fetal ultrasound images acquired following a blind obstetric sweep protocol [(DeStigter et al, 2011)](https://doi.org/https://doi.org/10.1109/GHTC.2011.39).

and writes the following output:
- Fetal abdomen segmentation mask to `"/output/images/fetal-abdomen-segmentation/output.mha"`: 2D segmentation mask of the fetal abdomen where 0-background and 1-fetal abdomen.
- Fetal abdomen ultrasound frame number at `"/output/fetal-abdomen-frame-number.json"`: Integer that represents the fetal abdomen ultrasound frame number corresponding to the output segmentation mask, indexed from zero. The value can also be -1, indicating that no frame was selected.

## Running inference 
The implementation of the algorithm inference is in `inference.py`. The code snippet below focuses exclusively on handling inputs and outputs, and executing the algorithm's inference process. It initializes the `FetalAbdomenSegmentation` algorithm and uses its `predict` and `postprocessing` methods to generate a stack fetal abdomen sementation masks. Following this, the `select_fetal_abdomen_mask_and_frame` function determines the best frame and corresponding fetal abdomen segmentation mask for fetal abdominal circumference measurement. The selected outputs are then saved using the `write_array_as_image_file` and `write_json_file` functions. 

    # Instantiate the algorithm
    algorithm = FetalAbdomenSegmentation()

    # Forward pass
    fetal_abdomen_probability_map = algorithm.predict(
        stacked_fetal_ultrasound_path, save_probabilities=True)

    # Postprocess the output
    fetal_abdomen_postprocessed = algorithm.postprocess(
        fetal_abdomen_probability_map)

    # Select the fetal abdomen mask and the corresponding frame number
    fetal_abdomen_segmentation, fetal_abdomen_frame_number = select_fetal_abdomen_mask_and_frame(
        fetal_abdomen_postprocessed)

    # Save your output
    write_array_as_image_file(
        location=OUTPUT_PATH / "images/fetal-abdomen-segmentation",
        array=fetal_abdomen_segmentation,
        frame_number=fetal_abdomen_frame_number,
    )
    write_json_file(
        location=OUTPUT_PATH / "fetal-abdomen-frame-number.json",
        content=fetal_abdomen_frame_number
    )
