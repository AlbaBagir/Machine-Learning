import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn import neural_network, naive_bayes, svm, neighbors
from sklearn.metrics import accuracy_score

#load data
data = pd.read_csv('dataset/phishing.csv')
label = data['class']
dataset = data.drop(['class'], axis=1)

xTrain, xTest, yTrain, yTest = train_test_split(dataset, label, test_size=0.2, random_state=0)

MLPClassifier = neural_network.MLPClassifier(alpha=0.00001)
MLPClassifier.fit(xTrain, yTrain)
neuralNetwork = MLPClassifier.predict(xTest)
print('Neural Network: ', accuracy_score(yTest, neuralNetwork))

gaussianNB = naive_bayes.GaussianNB()
gaussianNB.fit(xTrain, yTrain)
naiveBayes = gaussianNB.predict(xTest)
print('Naive Bayes: ', accuracy_score(yTest, naiveBayes))


LinearSVC = svm.LinearSVC(C=0.01)
LinearSVC.fit(xTrain, yTrain)
linearSVC = LinearSVC.predict(xTest)
print('Linear SVC: ', accuracy_score(yTest, linearSVC))

pipline = neighbors.KNeighborsClassifier(n_neighbors=5)
pipline.fit(xTrain, yTrain)
pipeline = pipline.predict(xTest)
print('Pipeline: ', accuracy_score(yTest, pipeline))

