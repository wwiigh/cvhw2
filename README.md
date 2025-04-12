# NYCU Computer Vision 2025 Sprint HW2
Student ID: 313553002  
Name: 蔡琮偉
## Introduction
The objective of the homework is Digit Recognition, use Faster R-CNN as backbone. To
finish the task, I choose fasterrcnn_resnet50_fpn as the backbone, with some data processing
to get better performance.
With the above method, the model get 0.38 at Task 1, and 77% at Task 2.
## How to install
### Install the environment
`
conda env create -f environment.yml
`  
If there are torch install error, please go to https://pytorch.org/get-started/locally/ to get correct torch version  
If get no tensorboard error in training, please run following command  
`
conda install -c conda-forge tensorboard
`  
or you can remove all SummaryWriter function  
### Pretained model and dataset download
Pretained model: [https://drive.google.com/file/d/1xz5ITdQQgvOm7fJItMF445Q0rcK1Idks/view?usp=sharing](https://drive.google.com/file/d/1uuW3LSQaPa5U9z07FKtPpHh7nkLkBIEZ/view?usp=sharing)  
dataset: [https://drive.google.com/file/d/1fx4Z6xl5b6r4UFkBrn5l0oPEIagZxQ5u/view?usp=drive_link](https://drive.google.com/file/d/13JXJ_hIdcloC63sS-vF3wFQLsUP1sMz5/view?usp=sharing)
### File structure
create model and data folder, put the model and unzip dataset to corresponding folder. It should look like this  
project-root/  
├── src/  
│   ├── dataset.py  
|   ├── model.py  
|   ├── test.py  
|   ├── train.py  
│   ├── utils.py  
├── data/       
│   ├── test/  
│   ├── train/  
│   └── val/  
├── model/  
│   ├── pretained.pth  
├── README.md          
├── environment.yml    
└── .gitignore          
### How to use
For test, change the function test's parameter to the pretained model path then run  
`
python src/test.py
`  
For training, run  
`
python src/train.py
`  
You can change the model backbone in model.py, just uncommit self.model to ResNet or MobileNet version.
## Performance snapshot
![image](https://github.com/user-attachments/assets/664e9aae-8625-41b7-b005-eaac6e8cdece)


