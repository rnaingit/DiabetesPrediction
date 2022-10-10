# DiabetesPrediction

1. Assignment Overview 

The goal of the assignment is to develop a Diabetes Prediction ML project. In the first part of the 
assignment, I have implemented Logistic regression and plotted the train loss vs validation loss and train 
accuracy vs validation accuracy graphical representation using Matplotlib library using the sample 
dataset provided to us.

2. About Logistic Regression
Definition: Logistic regression is a supervised learning classification algorithm used to predict the 
probability of a target variable. The nature of target or dependent variable is dichotomous, which means 
there would be only two possible classes.
In simple words, the dependent variable is binary in nature having data coded as either 1 (stands for 
success/yes) or 0 (stands for failure/no).
Mathematically, a logistic regression model predicts P(Y=1) as a function of X. It is one of the simplest 
ML algorithms that can be used for various classification problems such as spam detection, Diabetes 
prediction, cancer detection etc.

3. DataSet
To implement the logistic regression models with diabetes data set with 768 samples. I have split the
data samples using model_selection.train_test_split from sklearn library. The splitting 
of data is in such a way that I kept 60% data for training and 20 % - 20% data for validation and testing 
respectively

4. Tools & Editor
I have used Juypter notebook for the implementation of the project.

5. Processing: 
The First part of the implementation is to import all the useful libraries for the project I have used sklearn 
to split the data pandas for making data frame and reading the csv file and numpy for all the nd array 
operations. The next step is to reshape the data such that features are along the rows and instances are 
along the columns in the split the data. There are 3 main functions the code.
1) Sigmoid function: 1 / (1 + np.exp(-x)) (Activation function)
2) Prediction Function: that takes Wight bias and dataset and passes it to Sigmoid function to 
make predictions.
3) Accuracy Function: To calculate the accuracy I have used
metrics.accuracy_score(Y_actual, Y_predicted)
Flow for the Logistic Regression:
For a feature x and output label y.
Z= wx+b
Where w is the weight associated with the feature x and b is the bias. Now this z (which will act 
as a input to sigmoid function)is passed to the activation function in our case its sigmoid 
function which will return us the prediction(true or false/ 1/0).
1 / (1 + np.exp(-input))
Now the loss for this epoch is calculated by the formula
Once the loss function is computed using this formula
We calculate dW and db and gradient descent algo is used with learning rate to find the new 
weights and bias.
 W = W - learning_rate * dW
 b = b - learning_rate * db
As the weights are updates we will plot graph between loss_validation and loss_training.
Similarly, we will find the training accuracy, validation accuracy and testing accuracy for each 
epoch and later use them to draw conclusion using matplotlib.


6. Please import the code to see the results.
