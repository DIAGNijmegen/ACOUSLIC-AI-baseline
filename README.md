# ACOUSLIC-AI baseline algorithm
The ACOUSLIC-AI baseline algorithm serves as an initial benchmark for the [ACOUSLIC-AI challenge](https://acouslic-ai.grand-challenge.org/) and a template to assist participants in correctly packaging their own algorithms.

## Managed by
[Diagnostic Image Analysis Group](https://diagnijmegen.nl/) and [Medical UltraSound Imaging Center](https://music.radboudimaging.nl/), Radboud University Medical Center, Nijmegen, the Netherlands.

## Contact information 
- Sofía Sappia: mariasofia.sappia@radboudumc.nl
- Keelin Murphy: keelin.murphy@radboudumc.nl

## Algorithm
This algorithm is hosted on [Grand-Challenge](https://grand-challenge.org/algorithms/acouslic-ai-baseline)

### Summary
This algorithm selects the best frame for measuring the fetal abdominal circumference on a series of blind-sweep ultrasound images. Together with this frame, it provides the corresponding fetal abdomen binary segmentation mask. 

### Mechanism
This algorithm is a deep learning-based classification and segmentation model based on the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework. It was trained for one fold with an 80/20 train/validation split on the ACOUSLIC-AI Training and Development Dataset. The code to run inference using this baseline method is in `inference.py`.

# How to get started?
Read these:
- [Setting up Docker on your local system](documentation/setting_up_docker.md)
- [Building your own AI algorithm](documentation/building-your-own-ai-algoritm.md)


