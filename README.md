# ACOUSLIC-AI baseline algorithm
The ACOUSLIC-AI baseline algorithm serves as an initial benchmark for the [ACOUSLIC-AI challenge](https://acouslic-ai.grand-challenge.org/) and a template to assist participants in correctly packaging their own algorithms.

## Managed by
[Diagnostic Image Analysis Group](https://diagnijmegen.nl/) and [Medical UltraSound Imaging Center](https://music.radboudimaging.nl/), Radboud University Medical Center, Nijmegen, the Netherlands.

## Contact information 
- Sofía Sappia: mariasofia.sappia@radboudumc.nl
- Keelin Murphy: keelin.murphy@radboudumc.nl

## Algorithm
This algorithm is hosted on [Grand-Challenge](https://grand-challenge.org/algorithms/acouslic-ai-baseline).

### Summary
This algorithm selects the best frame for measuring the fetal abdominal circumference on a series of blind-sweep d B-mode ultrasound images. Together with this frame, it provides the corresponding fetal abdomen binary segmentation mask. 

### Mechanism
This algorithm is a deep learning-based classification and segmentation model based on the [nnUNet framework](https://github.com/MIC-DKFZ/nnUNet). It was trained for one fold with an 80/20 train/validation split on the ACOUSLIC-AI [Training and Development Dataset](https://doi.org/10.5281/zenodo.11005384). The code to run inference using this baseline method is in `inference.py`. For more details on this algorith, please refer to [An overview of the baseline algorithm](documentation/overview-baseline-algorithm.md).

# How to get started?
### Cloning the repository to your local system
We recommend that you begin by cloning this repository and packaging it into a Docker container. Once packaged, upload it as a [Grand-Challenge algorithm](https://grand-challenge.org/algorithms/) and submit it to the [Preliminary Development Phase](https://acouslic-ai.grand-challenge.org/evaluation/preliminary-development-phase/submissions/create/) of the challenge. This will help you become acquainted with the platform and ensure you can create a successful submission. Follow these steps to clone the repository to your local system:
1. Open a new terminal or command window.
2. Navigate to the location where you want to clone the repository by typing: \
    ```cd /path/to/your/desired/location```
3. Clone the repository with the following command: \
    ```git clone https://github.com/DIAGNijmegen/ACOUSLIC-AI-baseline.git ```
### Getting started with the algorithm template
Explore the following resources to understand the baseline algorithm, set up your environment, and begin customizing your own AI solution:
- **Overview of the baseline algorithm:** Start by reading [this document](documentation/overview-baseline-algorithm.md) to get a detailed understanding of the baseline algorithm provided.
- **Setting up Docker locally:** Ensure you have Docker configured on your system by following the instructions in [this guide](documentation/setting_up_docker.md).
- **Building your own AI algorithm:** Learn how to develop and customize your own AI algorithm by reviewing [this document](documentation/building-your-own-ai-algorithm.md).

# Issues
Please feel free to report any issues you encounter [here](https://github.com/DIAGNijmegen/ACOUSLIC-AI-baseline/issues). 


