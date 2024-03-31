# End To End Deep Learning Project

This repository contains solutions for various particle physics tasks using deep learning models. Below are the details of each task along with their respective implementations.

## Common Task 1: Electron/Photon Classification

### Dataset Description
- Two types of particles: electrons and photons
- Images are represented as 32x32 matrices with two channels: hit energy and time

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20Task%201/ResNetCBAM.png" width="350" title="hover text">
</p>

### Solution
- Implemented a ResNet-15 model for classification
- I used ensembling of two Resnet-15 models, one for learning pixel distribution of hit energy channel and other of time channel.
- Trained the model using K-Fold Cross Validation. I trained the model for 5 folds.
- Ensured no overfitting on the test dataset

#### Training Notebook:
Here are the notebooks showing complete training process

- [Common Task 1.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20Task%201/Common%20Task1%20Approach%201/common-task-1.ipynb)
- Includes data loading, model definition, training, evaluation, and model weights

#### Example Notebook:
These are Example Notebooks to inference or reproduce the results

-  [Example Test.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20Task%201/Common%20Task1%20Approach%201/Example%20Test.ipynb)
-  [Example Train.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20Task%201/Common%20Task1%20Approach%201/Example%20Train.ipynb)

## Common Task 2: Deep Learning based Quark-Gluon Classification

### Dataset Description
- Two classes of particles: quarks and gluons
- Images are represented as 125x125 matrices in three-channel images

<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/1_B_ZaaaBg2njhp8SThjCufA.png" width="350" title="hover text">
</p>


<p align="center">
  <img src="https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/VisionTransfomer.png" width="350" title="hover text">
</p>

### Solutions
1. VGG with 12 layers
2. Vision Transformer

### Implementation
- Implemented both models for classification
- Trained the model using K-Fold Cross Validation(5 Folds is performed).
- Ensured no overfitting on the test dataset

#### Notebooks:
Here are the notebooks showing complete training process

- [vgg-task2.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/vgg-task2.ipynb)
- [VisionTransformer - Task 2.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/VisionTransformer%20-%20Task%202.ipynb)
- Include data loading, model definition, training, evaluation, and model weights

#### Example Notebook:
These are Example Notebooks to inference or reproduce the results

-  [Example Test vgg12.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/Example_test_vgg.ipynb)
-  [Example Test VIT.ipynb](https://github.com/Wodlfvllf/End-to-End-Deep-Learning-Project/blob/main/Common%20task2/Example_test_vit.ipynb)

## Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction
Trained a Resnet-15 model using SimCLR self-supervised learning approach on unlabelled approach.
Finetuned the Encoder backbone on labelled dataset.
Trained the Resnet-15 encoder backbone from scratch.

<p align="center">
  <img src="https://github.com/Wodlfvllf/-Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction/blob/main/Specific%20task%203c%20Self-Supervised%20Learning/simclr-general-architecture.png" width="350" title="hover text">
</p>

### Comparison of With and Without Pretrained Vision Transformer Model

                          | Model               | Accuracy |
                          |---------------------|----------|
                          | With Pretrained     | 0.7589   |
                          | Without Pretrained  | 0.790    |

### Notebooks
-[Resnet-15 Pretraining using SimCLR](https://github.com/Wodlfvllf/-Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction/blob/main/Specific%20task%203c%20Self-Supervised%20Learning/SIMCLR%20Pretraining.ipynb)
-[Resnet-15 Finetuned using SimCLR](https://github.com/Wodlfvllf/-Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction/blob/main/Specific%20task%203c%20Self-Supervised%20Learning/simclr-finetune.ipynb)
-[Resnet-15 Scratch](https://github.com/Wodlfvllf/-Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction/blob/main/Specific%20task%203c%20Self-Supervised%20Learning/Resnet-15%20scratch.ipynb)

### Example Notebooks
-[Resnet 15 scratch Example test](https://github.com/Wodlfvllf/-Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction/blob/main/Specific%20task%203c%20Self-Supervised%20Learning/Example%20test%20resnet.ipynb)
-[Resnet 15 SimCLR Finetuned Example Test](https://github.com/Wodlfvllf/-Self-Supervised-Learning-for-End-to-End-Particle-Reconstruction/blob/main/Specific%20task%203c%20Self-Supervised%20Learning/example-sim-clr-finetuned.ipynb)

## Dependencies
- Python 3.x
- Jupyter Notebook
- PyTorch
- NumPy
- Pandas
- Matplotlib

Install these dependencies using pip or conda.
