<h1>Fashion MNIST Neural Network Classifier</h1>

<h2>Description</h2>
A neural network image classifier built using TensorFlow and Python. This project trains a neural network to classify images from the Fashion MNIST dataset into one of 10 classes, such as T-shirts, trousers, and shoes. The model achieves a test accuracy of ~88% and visualizes predictions and performance metrics.
<br />

<h2>Features</h2>
The image classifier can:
- Preprocess grayscale images from the Fashion MNIST dataset.
- Train a deep learning model using dropout regularization for improved generalization.
- Predict the class of unseen test images with confidence scores.
- Visualize predictions with corresponding bar charts for confidence scores.
- Analyze performance using a confusion matrix.
<br />

<h2>Languages and Libraries Used</h2>
- <b>Python</b>
- <b>TensorFlow</b>
- <b>NumPy</b>
- <b>Matplotlib</b>
- <b>Seaborn</b>
- <b>Scikit-learn</b>

<h2>Environment Used</h2>
- <b>Spyder</b> (for development and testing)
- <b>Command Line</b> (for running the script)

<h2>Program Walk-Through</h2>

<p align="center">
<h3> Dataset Loading and Preprocessing</h3>
The dataset is loaded using TensorFlow's `fashion_mnist` API. Images are normalized to ensure consistent scaling for the neural network. The dataset includes 60,000 training images and 10,000 test images of 10 different clothing categories. <br/>
<img src="https://i.imgur.com/iga7VY4.png" height="80%" width="80%" alt="Dataset Loading and Preprocessing"/>
<br />
<br />
Dataset Overview (Sample Image and Labels): <br/>
<img src="https://i.imgur.com/lOkEZe3.png" height="80%" width="80%" alt="Dataset Overview"/>
<br />
<br />
<h3>2. Neural Network Architecture</h3>
The neural network consists of a flatten layer, two dense layers with ReLU activation, and dropout layers to prevent overfitting. The output layer has 10 neurons (one for each class). <br/>
<img src="https://i.imgur.com/THt9H0z.png" height="80%" width="80%" alt="Neural Network Architecture"/>
<br />
<br />

<h3>3. Model Training</h3>
The model is trained for 10 epochs using the Adam optimizer and sparse categorical crossentropy loss. The training accuracy improves over epochs. <br/>
<img src="https://i.imgur.com/h3NHnnc.png" height="80%" width="80%" alt="Model Training"/>
<br />
<br />

<h3>4. Model Evaluation</h3>
The model is evaluated on the test dataset, achieving approximately 88% accuracy. <br/>
<img src="https://i.imgur.com/Oq7YwSF.png" height="80%" width="80%" alt="Model Evaluation"/>
<br />
<br />

<h3>5. Predictions Visualization</h3>
The program visualizes predictions for multiple test images. For each image, the left side shows the image and the predicted label, while the right side shows a bar chart of probabilities for all classes. <br/>
<img src="https://i.imgur.com/Sk91Xge.png" height="80%" width="80%" alt="Predictions Visualization"/>
<br />
<br />

<h3>6. Confusion Matrix</h3>
The confusion matrix provides an overview of the model's classification performance, highlighting correct and incorrect predictions. This helps identify patterns where the model struggles. <br/>
<img src="https://i.imgur.com/iINkT1f.png" height="80%" width="80%" alt="Confusion Matrix"/>
<br />
</p>

