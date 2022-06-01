# Abstract

Developed a **CNN+LSTM** based non-sequential model that can describe real-life images. <br>
Utilized a **CNN** layer to process images and a Glove.6B.50D word embedding layers to 
encode words. <br>
Finally used an **LSTM** layer for generating descriptions in a sequential manner.


# Technology Used

**Language**: Python <br>
**Neural Networks**: Convolutional Neural Networks(CNN), Recurrent Neural Networks(RNN) and Long Short-term Memory Networks(LSTM) <br>
**Algorithms**: Gradient Descent with backpropagation <br>
**Python Notebook**: Google Collab Notebook <br>
**Libraries**: Keras, Tensorflow, Pandas, Numpy, Matplotlib, CV2, Pickle <br>
**Transfer Learning**: ResNet50, Glove 6B 50D word embeddings <br>
**Dataset**: Flickr8K dataset <br>
**Web Framework**: Flask


# Methodology

1. The model is based on a typical **encoder-decoder** architecture.
2. A deep Resnet-based CNN model and a Glove.6B.50D word embedding model layer are first used to encode images and texts respectively.
3. Then the output of both the layers is combined and given as input to the decoder.
4. An LSTM layer is then used as a decoder, which generates a caption for the input image in a sequential manner.


# Implementation Steps

![image](https://user-images.githubusercontent.com/65055067/171406717-0d0093a5-6eda-47cc-9744-b38cf1268fc4.png)
<br> <p align="center"> **_Model Lifecycle_**</p> <br>

1. Data Collection
2. Understanding the data
3. Data Cleaning
4. Loading the training dataset
5. Visual Data Preprocessing - Images
6. Textual Data Preprocessing - Captions
7. Data Preparation using Generator Function
8. Word Embeddings
9. Model Architecture
10. Inference


# Result Screenshots

![image](https://user-images.githubusercontent.com/65055067/170859021-a84baa21-ec6c-4e00-93bc-222e590f3d5d.png)
<br> <p align="center"> _Screenshot 1_</p> <br>
![image](https://user-images.githubusercontent.com/65055067/170859036-e81c4388-22e1-4529-98c9-3b98eb7b1dc2.png)
<br> <p align="center"> _Screenshot 2_</p> <br>


# Future Scope

1. Self-Driving Cars - If we can properly capture the scene around the self-driving car, it can contribute to the self-driving car.
2. Aid to the blind - We can create a product for the blind, which can guide them by first converting the scene into text and then the text to voice.
3. CCTV cameras - If somehow we can use the captioning model with live images captured by CCTV cameras, then it can raise an alarm as soon as it captures some malicious activity. This could help reduce some crimes and/or accidents.


# References

1. [Automatic image annotation](https://en.wikipedia.org/wiki/Automatic_image_annotation)
2. [Deep Visual-Semantic Alignments for Generating Image Descriptions](https://arxiv.org/pdf/1412.2306.pdf)
3. [Show and Tell: A Neural Image Caption Generator](https://arxiv.org/abs/1411.4555)
4. [Where to put the Image in an Image Caption Generator](https://arxiv.org/pdf/1703.09137.pdf)
