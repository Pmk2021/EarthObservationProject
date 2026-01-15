# EarthObservationProject

This README provides an overview of the code for the project "Hurricane Damage Detection with Deep Learning" for the class "ENV-540 - Image Processing for Earth Observation". 

## Project Summary
For this project, we trained and evaluated three models: a ResNet, a LeVit, and a Dino. We then chose the model that performed the best for calibration and prediction on the final test set. 

## Structure of Github repo

### Folders: 
- LeVit: This folder contains 
    - the jupyter notebook IPEO project LeVit.ipynb which loads the data, preproccesses it, trains the LeVit model, and evaluates it. 
    - a folder called levit_hurricane_damage which contains the trained model parameters
    - a file called requirement.txt which contains all installed packages to run the code in this folder
- ResNet: This folder contains
    - a file called dataset.py which loads the dataset
    - a file called IPEO_project_ResNet which preprocesses the data, trains the ResNet model, and evaluates it.
    - a file called best_resnet18 which contains the trained model parameters
- Dino: This folder contains: 
    - a file called dataset.py wich loads the data
    - a file called models.py which defines the model
    - a file called train.py which trains the model
    - a file called dino_classifier.pth which contains the trained model parameters
- sample_test_set: This folder contains two subfolders (damage and no_damage) which each contain one sample image labelled as damaged and not damaged respectively. These sample images were used in the inference.ipynb as requested by course instructions.

### Files: 

- inference.ipynb: This file loads two sample images, loads the trained parameters from out best model, i.e., the DINO, runs the inference, and displays the prediction together with the calibrated probability. 
- IPEO project.ipynb which loads the data, preprocesses it, evaluates the three models previously trained, and calibrates the predicted probabilites of the model with the highest validation accuracy, i.e., the DINO. 
- envrionment.yaml: This file lists all our python packages used. 

### Important!
The trained parameters of the model used in inference.ipynb are saved in the Dino folder: "Dino/dino_classifier.pth"


