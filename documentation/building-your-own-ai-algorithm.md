# Building your own AI algorithm: Step-by-step instructions
The instructions below provide guidelines for adapting the provided baseline code to package your own AI algorithm for deployment as a Docker container in the [ACOUSLIC-AI challenge](https://acouslic-ai.grand-challenge.org/).

### 1. Setting Up Your Development Environment
Ensure you have the necessary software installed:
- Python 
- Docker
- Necessary Python libraries: numpy, SimpleITK, torch, glob2, and any other libraries your algorithm requires.

### 2. Integrating Your Algorithm
Locate the section in the code where the FetalAbdomenSegmentation class is instantiated and used. You will replace this segment with your own model’s code.
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
Here is what you need to do:
- `YourCustomAlgorithm`: Replace this placeholder with your class that encapsulates your algorithm's functionality.
- `predict`: This method should accept a loaded image and return a prediction map. Adjust this based on whether your algorithm outputs probabilities or direct segmentations.
- `postprocess`: Implement any necessary steps here to refine the prediction into a usable segmentation mask.
- `extract_segmentation`: Implement logic to extract the segmentation mask and identify the relevant frame number from the postprocessed output.

This is a suggested pipeline for your reference. Feel free to modify it according to your specific needs or preferences.

### 3. Handling Input and Output
Make sure your implementation adheres to the input and output conventions set by the provided code:
- **Input:** Your algorithm may use the load_image_file_as_array() function to load images.
- **Output:** Your algorithm should output two key pieces of information:
  - Segmentation Mask: A 2D numpy array of type `np.uint8`. This output should be saved to `location=OUTPUT_PATH / "images/fetal-abdomen-segmentation"` using the `write_array_as_image_file`. Note that you will need to specify the frame number as an additional argument.
  - Frame number: An integer indicating the frame number where the segmentation was found, or -1 if no relevant frame was found. This output should be saved to `location=OUTPUT_PATH / "fetal-abdomen-frame-number.json"` using the `write_json_file`. 

It is highly advisable to use these functions and maintain the specified output locations without modifications. Doing so ensures that outputs are saved in accordance with the expected internal workings of the Grand Challenge platform, facilitating smooth integration and functionality.
  

### 4. Packaging and Testing Your Container
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

### 5. Exporting the Container for Deployment
To prepare your Docker image for deployment, save it as a compressed file:
```
docker save your-algorithm-container | gzip -c > your-algorithm-container.tar.gz
```

### Conclusion
These steps guide you through customizing the provided framework to suit your algorithm, ensuring it can be packaged and deployed effectively in [Grand-Challenge.org](https://acouslic-ai.grand-challenge.org/). Ensure your algorithm complies with the input and output specifications for seamless integration and testing.
