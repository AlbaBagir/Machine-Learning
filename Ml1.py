import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neural_network, naive_bayes, svm, neighbors
from sklearn.metrics import accuracy_score

#read the data
data = pd.read_csv('dataset/phishing.csv')
label = data['class']   #get the label  
dataset = data.drop(['class'], axis=1)  #exclude the label

xTrain, xTest, yTrain, yTest = train_test_split(dataset, label, test_size=0.2, random_state=0)  #split the data into training and testing

MLPClassifier = neural_network.MLPClassifier(alpha=0.00001)
MLPClassifier.fit(xTrain, yTrain)
trainMlp = MLPClassifier.predict(xTrain)
print("Training data accuracy: ", accuracy_score(yTrain, trainMlp))
mlpClassifier = MLPClassifier.predict(xTest)
print("MLPClassifier accuracy: ", accuracy_score(yTest, mlpClassifier))

gaussianNB = naive_bayes.GaussianNB()
gaussianNB.fit(xTrain, yTrain)
trainGaussian = gaussianNB.predict(xTrain)
print("Training data accuracy: ", accuracy_score(yTrain, trainGaussian))
gaussianClassifier = gaussianNB.predict(xTest)
print("GaussianNB accuracy: ", accuracy_score(yTest, gaussianClassifier))

LinearSVC = svm.LinearSVC(C=0.01)
LinearSVC.fit(xTrain, yTrain)
trainLinear = LinearSVC.predict(xTrain)
print("Training data accuracy: ", accuracy_score(yTrain, trainLinear))
LinearSVCClassifier = LinearSVC.predict(xTest)
print("LinearSVC accuracy: ", accuracy_score(yTest, LinearSVCClassifier))

KNNeighbors = neighbors.KNeighborsClassifier(n_neighbors=5)
KNNeighbors.fit(xTrain, yTrain)
trainKNN = KNNeighbors.predict(xTrain)
print("Training data accuracy: ", accuracy_score(yTrain, trainKNN))
KNNClassifier = KNNeighbors.predict(xTest)
print("KNNeighbors accuracy: ", accuracy_score(yTest, KNNClassifier))
