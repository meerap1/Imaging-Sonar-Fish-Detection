# Imaging-Sonar-Fish-Detection
This GitHub repository is designed for detecting fish in imaging sonar.
![image](https://github.com/meerap1/Imaging-Sonar-Fish-Detection/assets/156745402/fc4c23b5-9221-45b5-93cf-d7af0ccedacb)

## Table of Content
1. Introduction
2. Data Source
3. Creating Virtual Environment
4. Setup Yolov7 Repository
5. Modify Yolov7 files
6. Train Yolov7 Model
7. Testing on Images and Videos
8. Results
9. Inferences

### Introduction
I'm excited to share with you my latest project: a YOLOv7 model for Imaging snar fish detection. Leveraging state-of-the-art deep learning techniques, this model is designed to accurately identify and localize fish in images. Utilizing Anaconda Prompt, I've crafted an environment where you can easily set up and run the model on your own machine.

### Data Source:
The data used in this repository for fish object detection in marine sonar videos, which is a primary data collected and annotated. A subset of over 1500 + images and labels from this dataset was selected for training purposes. This curated dataset encompasses both positive and negative images, where positive images contain instances of fish, annotated with bounding boxes and class labels, while negative images depict marine scenes without fish objects. <br/>
<br/>
Furthermore, the dataset organization involves dividing the images and labels into train and val folders. Within each of these folders, there are two subfolders: one containing images and the other containing corresponding labels. <br/>
<br/>
### Creating Virtual Environement:
To create a virtual environment, open Anaconda Prompt and use the following code: <br/>
<br/>
`conda create -n yolov7_custom python=3.9` <br/>
<br/>
After hitting enter, type 'Y' and hit enter again. Once the environment is created, activate it with the command: <br/>
<br/>
`conda activate yolov7_custom` <br/>
<br/>
###  Setup Yolov7 Repository
Download the official YOLOv7 repository from the [link](https://github.com/WongKinYiu/yolov7) <br/>
<br/>
Download the zip file into your **yolov7_custom** folder. Once downloaded, unzip it and rename the folder to **yolov7_custom**. Then, move the folder to the main directory **yolov7_custom**and delete the empty directory. <br/>
<br/>
Inside the extracted repository, open the **requirements.txt** file. Check the Torch version; it should be greater than or equal to 1.7 but not 1.12. Similarly, the TorchVision version should be greater than or equal to 0.8.1 but not 0.13. Make note of these versions and visit the PyTorch website to install PyTorch with CUDA support, meeting the requirements listed in the txt files. <br/>
<br/>
![Screenshot 2024-04-16 115159](https://github.com/meerap1/FISH-DETECTION/assets/156745402/b0e2bf66-3340-48e4-bc3a-0f1cf753b797) <br/>
<br/>
Remove the lines referencing Torch and TorchVision from the 'requirements.txt' file, then save the file. Additionally, create a new text file and paste the pip install command used to install PyTorch with CUDA support. Ensure the command is correctly formatted, like so: <br/>
![Screenshot 2024-04-16 120410](https://github.com/meerap1/FISH-DETECTION/assets/156745402/d0187060-3507-4bc9-a362-e672b4788189)  <br/>
 <br/>
Save the file as **requirements_gpu.txt** <br/>
<br/>
Now, in Anaconda Prompt, navigate to the **yolov7_custom** directory and execute the command: <br/>
<br/>
`pip install -r requirements.txt` <br/>
<br/>
this will install all the libraries that are part of the requirements.txt once it is done execute another command <br/>
<br/>
`pip install -r requirements_gpu.txt` <br/>
<br/>
This will install PyTorch with CUDA support. <br/>
<br/>
### Modify Yolov7 Files
Now, the **train** and **validation** folders should be moved into the **data** folder inside the **yolov7_custom** directory. <br/>
<br/>
Now make a copy of coco.yaml file in the data folder and rename it to **custom_data.yaml** . open it and modify like below <br/>
<br/>
In the text file, we will modify the **train** and **val** paths. Also, **nc** represents the number of classes. In this case, we have only one class, which is **Fish**, and it is mentioned in the **names**. <br/>
<br/>
![Screenshot 2024-04-16 123401](https://github.com/meerap1/FISH-DETECTION/assets/156745402/0a6a70c9-79bf-41fa-bbb9-e4737eeaca06) <br/>
<br/>
Now, open the **training** folder in the **cfg** folder. Inside, you'll find 7 configuration files. You can choose any of them to train the fish dataset. In this case, we are selecting **yolov7.yaml**. Make a copy of this file. In the copied file, change only **nc** to 1 as we are having only one class, and then save it. <br/>
<br/>
![Screenshot 2024-04-16 130749](https://github.com/meerap1/FISH-DETECTION/assets/156745402/53cfd46b-5c8b-40e5-ba79-932575a57e0a) <br/>
<br/>
Now, we need to download the weights for the YOLOv7 base model from the official YOLOv7 repository. It is hidden in the [releases](https://github.com/WongKinYiu/yolov7/releases). Here, I downloaded **yolov7.pt** and copied it to the yolov7_custom directory. <br/>
<br/>
### Train Yolov7 Model
Open Anaconda Prompt, ensure you're in the **yolov7_custom** directory, and run the following command: <br/>
<br/>
`python train.py --workers 1 --device 0 --batch-size 8 --epochs 100 --img 640 640 --data data/custom_data.yaml --hyp data/hyp.scratch.custom.yaml --cfg cfg/training/yolov7_custom.yaml --name yolov7_custom --weights yolov7.pt` <br/>
<br/>
When the training is finished, you can find the training files in the yolov7_custom folder from runsin the main yolov7_custom directory. Additionally, within that directory, you'll find the 'weights' folder. Inside the 'weights' folder, locate the **best.pt** file, which represents the best weights based on the validation loss. Copy and paste this file to the main directory of 'yolov7_custom' and rename it to 'yolov7_custom.pt'.
### Testing on Images and Videos
Copy a test image and video and move them to the main directory 'yolov7_custom'. Rename the image to **1.jpg** and the video to **1.mp4**. Then, run the following command: <br/>
<br/>
`python detect.py --weights yolov7_custom.pt --conf 0.5 --img-size 640 --source 1.jpg --view-img --no-trace` <br/>
<br/>
`python detect.py --weights yolov7_custom.pt --conf 0.5 --img-size 640 --source 1.mp4 --view-img --no-trace` <br/>
<br/>
The output will be saved in the 'exp' folder within the 'detect' directory under 'runs'.
### Results
In my case, no detections are made in the images, whereas fish are getting detected in the videos.
### Inferences
It appears that not all fish are being detected accurately. Further work is needed to improve the detection performance. In conclusion, while the model shows promise in detecting fish in videos, refinement is required to achieve consistent and reliable detection across various scenarios and types of images.



