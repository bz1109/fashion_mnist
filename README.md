# fashion_mnist
Fashion MNIST Machine Learning Classification ( CNN and SVM )

### About the data:

The Fashion MNIST training set contains 60,000 examples, and the test set contains 10,000examples. Each image is 28 pixels in height and 28 pixels in width, for a total of 784 pixels in total. Each pixel has a single pixel-value associated with it, indicating the lightness or darkness of that pixel, with higher numbers meaning darker. This pixel-value is an integer between 0 and 255. The training and test data sets have 785 columns. The first column consists of the class labels (see above) and represents the article of clothing. The rest of the columns contain the pixel-values of the associated image.  Fashion MNIST also shares the same train-test-split structure as MNIST, for ease of use. 

### Objectives

    1-	Implement the following two methods to classify clothing items:
      i.	Keras Convolutional Neural Network
      ii.	Support Vector Machines using Scikit-Learn 	

    2-	Answering the following questions:
      i.	 What is the accuracy of each method?
      ii.	 What are the trade-offs of each approach?
      iii. What is the compute performance of each approach?

### Data preprocessing

    •	Reshaped the columns from (784) to (28, 28, 1). 
    •	Data is normalized by dividing the pixel values by 255 which results in values between 0 and 1.
    •	Saved label (target variable) feature as a separate vector.
    •	Split the train set into train and validation sets. 
    •	The validation set will be 20% from the original train set.
    •	New datasets: 
        o	Train data: 48,000 records 
        o	Validation: 12,000 records
        o	Test: 10,000 records

### Data Modeling:

#### Convolutional Neural Network (CNN) using Keras:
A convolution layer multiplies a matrix of pixels with a filter matrix or ‘kernel’ and sums up the multiplication values. Then the convolution slides over to the next pixel and repeats the same process until all the image pixels have been covered. 

A model using Keras python library is developed to predict the labels with the following criteria:
Sequential Model:

      •	Conv2D: 2D Convolutional layer:
          	filters = 32
          	kernel_size = (3 x 3)
          	activation = RELU
          	input_shape = 28 x 28
          	The input and output of the Conv2D is a 4D tensor
      •	MaxPooling2D is a Max pooling operation for spatial data:
          	pool_size = (2,2)
          	Conv2D with the following parameters:
              o	filters: 64
              o	kernel_size : (3 x 3)
              o	activation : RELU
      •	MaxPooling2D:
          	pool_size : (2,2)
      •	Conv2D:
          	filters: 128
          	kernel_size : (3 x 3)
          	activation : RELU
      •	Flatten:
      •	Dense, fully-connected NN layer:
          	units = 128;
          	activation: RELU
      •	Dense, output layer (fully connected):
          	units = 10 Classes;
          	activation = Softmax (standard for multiclass classification) 
      
The model is compiled using categorical cross-entropy and the Adam optimizer. The number if epochs used to fit the model is 50. To avoid overfitting, multiple drop-out layers are added to the model and the training is executed again. 

#### Support vector Machines Model (SVM):

A SVM classification model was built using Scikit-Learn library. SVM’s are highly effective in high dimensional spaces such as the case of fashion-MNIST dataset. Because different kernel functions can be specified for the decision function, SVM’s are versatile.  The disadvantage of SVM is that it doe not provide a probability estimate for classifications. For this model we used the default kernel ‘RGF’ function with a gamma coefficient of 0.05 and a penalty parameter C=5. 

### Findings:

The Convolutional Neural Network or CNN model developed with Keras (python API to TensorFlow) delivered a higher accuracy of 93% compared to the Support Vector Machine classifier using Scikit-Learn which reached a 91% accuracy. With both models struggling to predict “Shirts” class. Although the only adjustment made to the models was adding dropout layers to the CNN which eliminated overfitting, the performance of the SVM was surprisingly very good because it took significantly less time to train for the same dataset. In general, SVM’s are known to deliver very good results for image classification (See MNIST benchmarks) when using gaussian and polynomial kernels. Large CNN models deliver state-of-the-art results—often better than human—however, these networks are resource-intensive and require GPU hardware for training.  Overall, this analysis has shown that both, CNN and SVM algorithms are useful to identify clothing items.

### Resources
Kaggle.com. (n.d.). Retrieved from Fashion MNIST An MNIST-like dataset: https://www.kaggle.com/zalando-research/fashionmnist
Tensorflow Hub Authors. (2018). Google Colab. Retrieved from Fashion MNIST with Keras and TPUs: https://colab.research.google.com/github/tensorflow/tpu/blob/master/tools/colab/fashion_mnist.ipynb#scrollTo=edfbxDDh2AEs
Tensorflow, Google Inc. (n.d.). Train your first neural network: basic classification. Retrieved from Tensorflow: https://www.tensorflow.org/tutorials/keras/basic_classification
Yann LeCun, C. C. (n.d.). yann.lecun.com. Retrieved from THE MNIST DATABASE of handwritten digits: http://yann.lecun.com/exdb/mnist/


