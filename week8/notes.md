## Week 8 Notes : Deep Learning

Practice jupyter notebooks: 
   1. [Deep Learning](week8-deep-learning.ipynb)

### Video 1 :  Fashion Classification Project

----

[![Intro / Fashion Classification Project ](https://img.youtube.com/vi/it1Lu7NmMpw/0.jpg)](https://www.youtube.com/watch?v=it1Lu7NmMpw&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=69)

- Multi-class classification problem using deep learning to classify images of fashion items
- Dataset used : [Clothing dataset small](https://github.com/alexeygrigorev/clothing-dataset-small) will be used
  - This has 10 classes
  - Data in this repo has train, validation and test folders
- **Important Note** : Neural Networks fundamentals are not covered in this course.
  - cs231n course from stanford is suggested for learning fundamentals of neural networks and deep learning
  - [cs231n](https://cs231n.github.io/)

### Video 2 :  TensorFlow and Keras 

----

[![TensorFlow and Keras](https://img.youtube.com/vi/R6o_CUmoN9Q/0.jpg)](https://www.youtube.com/watch?v=R6o_CUmoN9Q&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=71)

- TensorFlow is the main deep learning framework used in this course
- Keras is an abstraction layer on top of TensorFlow that makes it easier to build and train neural networks
- Installed via requirements.txt file (`uv add -r requirements.txt`)
- Images can be loaded using `load_img` from `tensorflow.keras.preprocessing.image`
- Image Representation
  - Images are represented as 3D arrays (height, width, color channels) and equiv is numpy arrays in python
  - array values range from 0 to 255
  - three channels being Red, Green, Blue (RGB)
  - Array shapes are (height, width, 3) e.g., (224, 224, 3)

### Video 3 : Pre-Trained Convolutional Neural Networks

----

[![Pre-Trained Convolutional Neural Networks](https://img.youtube.com/vi/qGDXEz-cr6M/0.jpg)](https://www.youtube.com/watch?v=qGDXEz-cr6M&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=71)

- In [Keras website](https://keras.io/api/applications) there are many pre-trained models available
  - These models are trained on ImageNet dataset
  - These models have tradeoff between size, accuracy and speed
  - As of now [ImageNet](https://image-net.org/download.php) has 1,281,167 training images, 50,000 validation images and 100,000 test images
  - The dataset spans 1000 object classes
- Following are used for predicting an image
  - Tensorflow package `tensorflow.keras.applications.xception` 
  - For loading the model `model = Xception(weights='imagenet', input_shape=(299,299,3))`
  - For preprocessing the image before prediction `image_array = preprocess_input(image_array)`
  - For decoding the predictions `decoded_prediction = decode_predictions(img_prediction)[0]`
- The model predicts a class that is not closer to the fashion classes we have. 
  - Hence we will build our own model using transfer learning on top of this pre-trained model.

### Video 4 :  Convolutional Neural Networks

----

[![Convolutional Neural Networks](https://img.youtube.com/vi/BN-fnYzbdc8/0.jpg)](https://www.youtube.com/watch?v=BN-fnYzbdc8&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=72)

- Filters
  - CNN contain filters that are small sized images e.g., 3x3 or 5x5
  - These filters are convolved over the input image to produce feature maps
  - As many feature maps are generated as the number of filters used
  - Each filter enables the neural network to learn specific features from the input image
  - As complex the filters are, more complex features can be learned by the network
  - Conv layers are applied in sequence to learn complex features
- The sequence is `Image -> Conv Layer -> Vector Representation -> Dense Layer -> Output`
  - Image : Input image
  - Conv Layer : Convolutional layers that extract features from the image
  - Vector Representation : Flattened representation of the feature maps
  - Dense Layer : Fully connected layers that learn patterns from the vector representation
  - Output : Final output layer that produces the class probabilities
- Classification Output  
  1. Binary classification (sigmoid activation) i.e., `g(x) = sigmoid(x.w)`
     - The summed weighted inputs are passed through a sigmoid activation function to produce a probability between 0 and 1
     - The probability indicates the likelihood of the input belonging to the positive class
  2. Multi-class classification (softmax activation)
     - Sigmoid is a generalization of softmax for multiple classes
     - The sequence is `vector representation -> Dense Layer -> Softmax Activation`
     - The softmax activation function converts the output of the dense layer into a probability distribution over multiple classes
     - Each class is assigned a probability, and the sum of all probabilities equals 1
- Pooling Layers
  - Pooling layers are used to reduce the spatial dimensions of the feature maps
  - This helps in reducing the number of parameters and computation in the network
  - Common pooling operations include max pooling and average pooling
  - Pooling layers help in making the model more robust to variations in the input image

### Video 5 :  Transfer Learning

----

[![Transfer Learning](https://img.youtube.com/vi/WKHylqfNmq4/0.jpg)](https://www.youtube.com/watch?v=WKHylqfNmq4&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=73&pp=iAQB)

- Convolutional layers of the pre-trained model are used but not the dense layers. This is the core idea behind transfer learning.
  - Use an already trained model as a base and add new dense layers on top of it for our specific classification task.
  - Smaller image sizes of  (150, 150) are used  for faster training

- Model creation
  - `ImageDataGenerator` is used for loading and augmenting images from the dataset
  - `flow_from_directory` method is used to load images from the directory structure
  - Both training and validation data generators are created

- Base Model
  - `Xception` model is loaded without the top dense layers using `include_top=False`
  - Model is marked as non-trainable using `base_model.trainable = False` to freeze the weights during training

- Network design is as follows:
    - `inputs (150, 150, 3) of batch 32` -> i.e., input images of size 150x150 with 3 color channels (RGB)
    - `base_model (5 x 5 x 2048) of batch 32` -> i.e., output from the pre-trained Xception model
    - `Vector 1D (2048) of batch 32` -> i.e., flattened vector representation of the feature maps
    - `Outputs (10 classes) of batch 32` i.e., final output layer with 10 classes for fashion classification

- When predicted this model does not perform well as the base model is not trained on fashion images of our dataset.
    - Hence we will train the model on our dataset.

- Training the model
  - Learning Rate : value of 0.001 is used and `Adam` optimizer is used
  - Loss Function : `sparse_categorical_crossentropy` is used for multi-class classification
    - `from_logits=False` is set as the output layer uses softmax activation
  - Model is trained using `model.fit(train_ds, epochs=10, validation_data=val_ds)`

- Training and Validation accuracies are plotted using matplotlib to visualize the model performance over epochs.

### Video 6 :  Adjusting the Learning Rate

----

[![Adjusting the Learning Rate](https://img.youtube.com/vi/2gPmRRGz0Hc/0.jpg)](https://www.youtube.com/watch?v=2gPmRRGz0Hc&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=74)

- A good analogy of learning rate is how fast a book reader reads and understands the content. The rate of reading affects comprehension.
- Learning Rate is a hyperparameter that controls how much to change the model weights during training.
- Model is trained for multiple learning rates to find the optimal learning rate.
- Model accuracy corresponding to different learning rates is plotted using matplotlib.
- From the plot, the best learning rate is chosen for training the final model by observing the learning rate where accuracy starts to improve significantly without causing instability.


### Video 7 :  Checkpointing

----

[![Checkpointing](https://img.youtube.com/vi/NRpGUx0o3Ps/0.jpg)](https://www.youtube.com/watch?v=NRpGUx0o3Ps&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=75)

- Checkpointing is a way of saving a model when certain conditions are met during training.
- While training the model, the best model could be saved based on validation accuracy and not at the end of all epochs.
- This is done using `ModelCheckpoint` callback from `tensorflow.keras.callbacks`
- The checkpoint callback is created using:
  `ModelCheckpoint('xception_v1_{epoch:02d}_{val_accuracy:.3f}.h5', save_best_only=True, monitor='val_accuracy', mode='max')`
- The checkpoint callback is passed to the `model.fit` method using the `callbacks` parameter.
- After training, the best model is loaded using `load_model` from `tensorflow.keras.models`
- The loaded model is evaluated on the test dataset to get the final test accuracy.
- This ensures that the best performing model during training is used for evaluation, rather than the last model after all epochs.
- This technique helps in preventing overfitting and ensures better generalization on unseen data.


### Video 8 :  Adding More Layers

----

[![Fine-Tuning](https://img.youtube.com/vi/bSRRrorvAZs/0.jpg)](https://www.youtube.com/watch?v=bSRRrorvAZs&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=76)

- Current NN architecture is as follows:
  - ```
    # define the model input
    inputs = keras.Input(shape=(150,150,3))
    
    # pass the inputs through the base model, training is False to avoid updating batch norm layers
    base = base_model(inputs, training=False)
    
    # global average pooling to get vectors from feature maps
    vectors = keras.layers.GlobalAveragePooling2D()(base)
    
    # final dense layer for classification and keep the raw logits
    outputs = keras.layers.Dense(10)(vectors)
    
    model = keras.Model(inputs, outputs)
    ``` 
- We can add a new layer between the vectors and the outputs to improve model performance.
  - This new layer can be a dense layer and see if it improves performance.
- The layer size is varied across 10, 100, 1000 units to see the effect on performance.

## Video 9 :  Regularization and Dropout

----

[![Regularization and Dropout](https://img.youtube.com/vi/74YmhVM6FTM/0.jpg)](https://www.youtube.com/watch?v=74YmhVM6FTM&list=PL3MmuxUbc_hIhxl5Ji8t4O6lPAOpHaCLR&index=77)

- Regularization is a technique used to prevent overfitting in machine learning models.
  - It adds a penalty to the loss function to discourage complex models that fit the training data too closely.
  - Common regularization techniques include L1 and L2 regularization, dropout,

- Dropout is a regularization technique used to prevent overfitting in neural networks. 
  - It is done by randomly setting a fraction of input units to 0 at each update during training time.
  - This effectively hides some features and forces the network to learn more robust features that are useful in conjunction with many different random subsets of the other neurons.