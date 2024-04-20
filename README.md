# ACOUSLIC-AI baseline algorithm

## Managed by
[Diagnostic Image Analysis Group](https://diagnijmegen.nl/), Radboud University Medical Center, Nijmegen, the Netherlands

## Contact information 
- Sofía Sappia: mariasofia.sappia@radboudumc.nl
- Keelin Murphy: keelin.murphy@radboudumc.nl

## Algorithm
This algorithm is hosted on [Grand-Challenge](https://grand-challenge.org/algorithms/acouslic-ai-baseline)

### Summary
This algorithm selects the best frame for measuring the fetal abdominal circumference on a series of blind-sweep ultrasound images. Together with this frame, it provides the corresponding fetal abdomen binary segmentation mask. 

### Mechanism
This algorithm is a deep learning-based classification and segmentation model based on the [nnUNet](https://github.com/MIC-DKFZ/nnUNet) framework. It was trained for one fold with an 80/20 train/validation split on the ACOUSLIC-AI Training and Development Dataset.

### Source code
- [Algorithm inference](https://github.com/DIAGNijmegen/ACOUSLIC-AI-baseline/blob/main/inference.py)

## Building your own AI algorithm: Step-by-step instructions
The instructions below provide guidelines for adapting the provided baseline code to package your own medical imaging algorithm for deployment as a Docker container in the [ACOUSLIC-AI challenge](https://acouslic-ai.grand-challenge.org/).

### 1. Setting up your development environment
Ensure you have the necessary software installed:
- Python 
- Docker
- Necessary Python libraries: numpy, SimpleITK, torch, glob2, and any other libraries your algorithm requires.
### 2. Integrating your algorithm
- Locate the section in the code where the FetalAbdomenSegmentation class is instantiated and used. You will replace this segment with your own model’s code. Here is what you need to do:
```
# Instantiate the algorithm
algorithm = YourCustomAlgorithm()

# Load the image from the specified path
input_image = load_image_file_as_array(location=your_image_path)

# Perform prediction using your algorithm
prediction_map = algorithm.predict(input_image)

# Postprocess the prediction to refine the results
postprocessed_output = algorithm.postprocess(prediction_map)

# Extract the final segmentation mask and the frame number
segmentation_mask, frame_number = algorithm.extract_segmentation(postprocessed_output)
```
- `YourCustomAlgorithm`: Replace this placeholder with your class that encapsulates your algorithm's functionality.
- `predict`: This method should accept a loaded image and return a prediction map. Adjust this based on whether your algorithm outputs probabilities or direct segmentations.
- `postprocess`: Implement any necessary steps here to refine the prediction into a usable segmentation mask.
- `extract_segmentation`: Implement logic to extract the segmentation mask and identify the relevant frame number from the postprocessed output.

### 3. Handling input and output
Make sure your implementation adheres to the input and output conventions set by the provided code:
- **Input:** Your algorithm may use the load_image_file_as_array() function to load images.
- **Output:** Your algorithm should output two key pieces of information:
  - Segmentation mask: A 2D numpy array of type np.uint8.
  - Frame number: An integer indicating the frame number where the segmentation was found, or -1 if no relevant frame was found.

### 4. Packaging and testing your container
Follow these steps to package your algorithm:
- Create a Dockerfile that sets up the environment, installs dependencies, and configures the script as the container's entry point.
- Build the Docker image:
```
docker build -t your-algorithm-container .
```
- Test the container locally by running:
```
./test_run.sh
```
Ensure this script is set up to properly mount the directories for input and output.

### 5. Exporting the container for deployment
To prepare your Docker image for deployment, save it as a compressed file:
```
docker save your-algorithm-container | gzip -c > your-algorithm-container.tar.gz
```

### Conclusion
These steps guide you through customizing the provided framework to suit your algorithm, ensuring it can be packaged and deployed effectively in [Grand-Challenge.org](https://acouslic-ai.grand-challenge.org/). Ensure your algorithm complies with the input and output specifications for seamless integration and testing.
