# Installing Docker on local system 🐳
You must have [Docker](https://docs.docker.com/get-docker/) installed and running on your system for the following steps to work. If you are using Windows, we recommend installing [Windows Subsystem for Linux (WSL)](https://docs.microsoft.com/en-us/windows/wsl/install). For more details, please watch the [official tutorial by Microsoft for installing WSL 2 with GPU support](https://www.youtube.com/watch?v=PdxXlZJiuxA).


## Building, testing and exporting algorithm container on local system

If you're using Linux/macOS, please use the shell scripts (`.sh` files) stated in all following commands. If you're using Windows, please use the batch scripts (`.bat` files) stated in all following commands.
Clone this template repository to your local system, with [Git LFS](https://git-lfs.github.com/) initialized (e.g. this can be done directly using [GitHub Desktop](https://desktop.github.com/)). Build and test your inference Docker container by running `build.bat`/`build.sh`, followed by `test.bat`/`test.sh`. Please note, that testing also runs a build, so running the build command separately is not necessary (if you are certain that everything is set up correctly). Testing will build the container and run it on images provided in the `./test/` folder. It will then check the output files (fetal abdomen segmentation mask and fetal abdomen frame number) produced by your algorithm, against those pre-computed and present in `./test/`. If all tests have completed successfully, then you're good to go. Export your container using `export.ba/export.sh`. Once complete, your container image should be ready as a single `.tar.gz` file.
For more details, we highly recommend completing this general tutorial on creating GC algorithms:
https://grand-challenge.org/documentation/create-your-own-algorithm/

## Uploading algorithm container to Grand Challenge
You should now be ready to upload your algorithm container as a Grand Challenge algorithm. Please follow the instructions provided [here](https://acouslic-ai.grand-challenge.org/ai-algorithm-submissions/#:~:text=Uploading%20algorithm%20container%20to%20Grand%20Challenge).

## Create containers for your own AI algorithm
Following these same steps, you can easily encapsulate your own AI algorithm in an inference container by altering one of our provided baseline algorithm templates. You can implement your own solution by editing the functions in `./inference.py`. Any additional imported packages should be added to `./requirements.txt`, and any additional files and folders should be explicitly copied through commands in `./Dockerfile`. To update your algorithm, you can simply test and export your new Docker container, after which you can update the container image for the GC algorithm that you created on grand-challenge.org with the new one. Please note, that your container will not have access to the internet when they are executed on grand-challenge.org, so all necessary model weights and resources must be encapsulated in your container image a priori. You can test whether this is true locally using the `--network=none` option of docker run.
If something doesn't work for you, please feel free to add a post regarding the same in the forum.

## Submission to Preliminary Development Phase - Validation and Tuning Leaderboard
Once you have your trained AI model uploaded as a fully-functional GC algorithm, you're now ready to make submissions to the [Preliminary Development Phase](https://acouslic-ai.grand-challenge.org/evaluation/preliminary-development-phase/submissions/create/) of the challenge! Navigate to the "Preliminary Development Phase Submissions" page, select your algorithm, and click "Save" to submit. If there are no errors and evaluation has completed successfully, your score will be up on the leaderboard (typically in less than 24 hours).

⚠️ Please double-check all rules to make sure that your submission is compliant. Invalid submissions will be removed and teams repeatedly violating any/multiple rules will be disqualified. Also, please try-out your uploaded GC algorithm with a sample case before making a full submission to the leaderboard.
