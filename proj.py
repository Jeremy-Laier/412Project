# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
import numpy as np
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt
from pylab import show
from sklearn.model_selection import cross_val_score
import pandas as pd
import time
from sklearn.decomposition import PCA, KernelPCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import StackingClassifier


# %%
importantColumns = ["Sex", "Age", "BodyweightKg", "Best3SquatKg", "Best3BenchKg", "Best3DeadliftKg","TotalKg", "Tested"]
                    
data = pd.read_csv("lift.csv", usecols=importantColumns, nrows=5000)
data = data.fillna(0)
data = data.sample(frac=1).reset_index(drop=True)


# %%
features = []
labels = []
for _, row in data.iterrows():
    test = row['Tested']
    sex = row['Sex']
    age = row['Age']
    weight = row['BodyweightKg']
    total = row['TotalKg']
    squat = row['Best3SquatKg']
    bench = row['Best3BenchKg']
    dead = row['Best3DeadliftKg']
    
    if (age < 15):
        continue
    if (weight <= 0):
        continue
    if (total <= 0):
        continue
    if (squat <= 0):
        continue
    if (bench <= 0):
        continue
    if (dead <= 0):
        continue
    
    if (sex == 'F'):
        sex = 0
    else:
        sex = 1

    if (test == "Yes"):
        labels.append(1)
    else:
        labels.append(0)

    features.append([weight, squat, bench, dead, total, age, sex])


# %%
numTrain = int(len(features)*0.2)

trainFeatures = features[:numTrain]
testFeatures = features[numTrain:]
trainLabels = labels[:numTrain]
testLabels = labels[numTrain:]


# %%
testing = np.asarray(trainFeatures)

colors = []

for i in trainLabels:
    if (i == 1):
        colors.append("b")
    else:
        colors.append("r")


# %%
#Find cross-validation errors for all c values from 0.1 -> 100

X2 = []
Y2 = []

for c in np.logspace(0.01,2, num=25):
    model = SVC(C=c, gamma = 'auto', kernel = 'linear')
    X2.append(c)
    Y2.append(cross_val_score(model, testing[:, 4:6], trainLabels, cv=10, n_jobs= -1).mean())


# %%
plt.figure(figsize=(7, 5))

plt.plot(X2, Y2, alpha = .9)
plt.xlabel('C values stepped logramithically')
plt.ylabel('Cross Validation Scores')
plt.title('SVM - Figure 3')
show()


# %%
euclidean = []
Knear = []
for k in range(1,49+1):
    if k % 2 == 0:
        continue
    else: 
        eu256 = cross_val_score(
            KNeighborsClassifier(n_neighbors=k, metric='euclidean', n_jobs=-1), 
            testing[:, 4:6], trainLabels, cv=10, n_jobs= -1).mean()

        euclidean.append(eu256)
        Knear.append(k)

## Visualize Results for Euclidean 256
#plot the points
# plt.scatter(testing[:, 4:5], testing[:, 5:6], s=50,c=colors)


# %%
plt.figure(figsize=(8, 5))

#plot the regions
plt.scatter(Knear,euclidean,s=50,alpha=.6)

#setup the axes
plt.xlabel("k-Nearest Neighbors")
plt.ylabel("Accuracy")
plt.title("Euclidean kNN - Figure 5")
show()


# %%
trainScores = []
testScores =  []
xlabels = []
for hiddenLayers in [1,2,5,10]:
    for numberNodes in [2,5,10,50,100]:
        nodes =[]
        for i in range(hiddenLayers):
            nodes.append(numberNodes)
        xlabels.append("{} - {}".format(hiddenLayers, numberNodes))        
        mlp = MLPClassifier(epsilon=0.001, max_iter=1000000, hidden_layer_sizes=(nodes), activation="relu", alpha=0).fit(trainFeatures, trainLabels)
        train = cross_val_score(mlp, trainFeatures, trainLabels, cv=10, n_jobs=-1).mean()
        test = mlp.score(testFeatures, testLabels)
        trainScores.append(train)
        testScores.append(test)


# %%
layers  = []
nodes= []
xlabels = []
for hiddenLayers in [1,2,5,10]:
    for numberNodes in [2,5,10,50,100]:
        nodes =[]
        for i in range(hiddenLayers):
            nodes.append(numberNodes)
        xlabels.append("{}-{}".format(hiddenLayers, numberNodes)) 


fig, ax = plt.subplots()

plt.figure(figsize=(12, 5))

plt.plot(trainScores, label="Training Scores")
plt.plot(xlabels, testScores, label="Test Scores")
plt.legend()
plt.xlabel("20 Models were used, 1-10 Hidden Layers, 2-100 Num Neurons Per Hidden Layer")
plt.ylabel("Model Accuracy")

plt.title("Accuracy of Multiple NN Models 0.001 Eta -- Testing and Training Scores")

show()


# %%
n_estimators = [1, 5, 10, 100, 1000, 10000]
errorVals = []

for estimator in n_estimators:
    clf = AdaBoostClassifier(n_estimators=estimator, base_estimator=DecisionTreeClassifier(max_depth=1, criterion="entropy"))
    errorVals.append(1 - cross_val_score(clf, trainFeatures, trainLabels, cv=10).mean())


# %%
err = []
for i in errorVals:
    err.append(1 - i)

plt.figure(figsize=(6, 5))
plt.plot(n_estimators, err, alpha=0.9)
plt.xscale('log')
plt.title('AdaBoost Model, Decision Tree Depth 1')
plt.xlabel('Number of Estimators')
plt.ylabel('Error values')
show()


# %%
# 1.a
# make decision tree model 
max_leaf_nodes = [5, 10, 15, 20, 30, 40, 50, 75, 100, 200, 500, 1000, 10000]
errorVals = []

for leaf_nodes in max_leaf_nodes:
    clf = DecisionTreeClassifier(criterion="entropy", max_leaf_nodes= leaf_nodes)
    errorVals.append(cross_val_score(clf, trainFeatures, trainLabels, cv=10).mean())

plt.figure(figsize=(6, 4))
plt.plot(max_leaf_nodes, errorVals, alpha=0.9)
plt.xscale('log')
plt.title('Figure 4.1 Decision Tree')
plt.xlabel('Max leaf nodes')
plt.ylabel('Accuracy')
show()


# %%
n_estimators = [1, 5, 10, 100, 1000, 10000]
errorVals = []

for estimator in n_estimators:
    clf = AdaBoostClassifier(n_estimators=estimator, base_estimator=DecisionTreeClassifier(max_depth=20, criterion="entropy"))
    errorVals.append(cross_val_score(clf, trainFeatures, trainLabels, cv=10).mean())


plt.figure(figsize=(6, 5))
plt.plot(n_estimators, errorVals, alpha=0.9)
plt.xscale('log')
plt.title('AdaBoost Model, Decision Tree Depth 20')
plt.xlabel('Number of Estimators')
plt.ylabel('Accuracy')
show()


# %%
svmModel = SVC(C=30, gamma = 'auto', kernel = 'linear').fit(trainFeatures, trainLabels)
nnModel = MLPClassifier(epsilon=0.0001, max_iter=10000, hidden_layer_sizes=([10]), activation="relu", alpha=0).fit(trainFeatures, trainLabels)
kNNModel = KNeighborsClassifier(n_neighbors=21, metric='euclidean', n_jobs=-1).fit(trainFeatures, trainLabels)

estimators = [('nn', nnModel), ('svm', svmModel), ('knn',kNNModel)]

clf = StackingClassifier(estimators=estimators, n_jobs=-1, cv=10).fit(trainFeatures, trainLabels)
print (str ( clf.score(testFeatures, testLabels)))


# %%
from math import sqrt
from math import log

def hoeffdingBound(model, sigLevel):
    n = len(testLabels)
    delta = 1 - sigLevel
    return  sqrt((1/(2* n) * log(2/delta)))

print (cross_val_score(clf, trainFeatures, trainLabels, cv=10, n_jobs=-1).mean())
print (clf.score(testFeatures, testLabels), hoeffdingBound(clf, 0.95))


# %%

print (cross_val_score(svmModel, trainFeatures, trainLabels, cv=10, n_jobs=-1).mean())
print (svmModel.score(testFeatures, testLabels), hoeffdingBound(svmModel, 0.95))


print (cross_val_score(nnModel, trainFeatures, trainLabels, cv=10, n_jobs=-1).mean())
print (nnModel.score(testFeatures, testLabels), hoeffdingBound(nnModel, 0.95))


print (cross_val_score(kNNModel, trainFeatures, trainLabels, cv=10, n_jobs=-1).mean())
print (kNNModel.score(testFeatures, testLabels), hoeffdingBound(kNNModel, 0.95))


# %%


