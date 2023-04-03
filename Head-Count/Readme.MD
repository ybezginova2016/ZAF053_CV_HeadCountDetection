## Crowd Counting using Transfer Learning
## Introduction
Crowd counting is the task of estimating the number of people in a given image or video. It has various real-world applications, such as in crowd management, event planning, and public safety. This project aims to develop a deep learning model for crowd counting using transfer learning.

## Dataset
The dataset used in this project is the Fmena14 crowd counting dataset, available on Kaggle here. It contains over 5,000 annotated images of crowds, captured from different perspectives and in different settings. Each image is labeled with the actual count of people in the image.
[LINK](https://www.kaggle.com/fmena14/crowd-counting)

## Preprocessing
The images in the dataset are of varying sizes and resolutions. To preprocess the images, they are resized to a fixed size of 256x256 pixels and normalized to have pixel values in the range [0, 1]. In addition, data augmentation techniques such as random cropping and horizontal flipping are used to increase the size of the training set and reduce overfitting.

## Models Used
Transfer learning is used to develop the crowd counting model. Specifically, the VGG-16 pre-trained on the ImageNet dataset is used as the base model. The fully connected layers of the VGG-16 model are removed and replaced with a new fully connected layer with a single output, representing the predicted count of people in the image. The weights of the pre-trained VGG-16 model are frozen during training, and only the weights of the new fully connected layer are updated.

## Training
The model is trained using the Adam optimizer with a learning rate of 0.001 and a batch size of 8. The training is done for 100 epochs, with early stopping used to prevent overfitting. The mean squared error (MSE) and mean absolute error (MAE) are used as the loss functions. The model is trained on a subset of 1,700 images from the dataset and validated on a subset of 300 images.

## Results
After training, the model achieves a validation MSE of 16.50 and a validation MAE of 3.19. The model is able to accurately estimate the count of people in images of crowds, with a small margin of error.
[Link to Saved model](https://drive.google.com/drive/folders/18jCyDWxYRUFhVUFSPsvGIluAJpo3Vw1N?usp=sharing)

## Usage in Real World
The trained model can be used to estimate the count of people in images or videos of crowds in real-world settings. The model can be integrated into existing systems for crowd management, event planning, and public safety.

## Papers
L. Zhang, M. Yang, and X. Feng, "Cross-scene crowd counting via deep convolutional neural networks," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2015, pp. 833–841.
[LINK](https://openaccess.thecvf.com/content_cvpr_2015/html/Zhang_Cross-Scene_Crowd_Counting_2015_CVPR_paper.html)
V. A. Sindagi and V. M. Patel, "CNN-based cascaded multi-task learning of high-level prior and density estimation for crowd counting," in Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, 2017, pp. 5245–5254. 
[LINK](https://openaccess.thecvf.com/content_cvpr_2017/html/Sindagi_CNN-Based_Cascaded_Multi-Task_CVPR_2017_paper.html)