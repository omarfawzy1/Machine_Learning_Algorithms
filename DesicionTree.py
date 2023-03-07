import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn.metrics import accuracy_score

# Read file
df = pd.read_csv("BankNote_Authentication.csv")

# Shuffling the data
df = df.reindex(np.random.permutation(df.index))

features = ['variance', 'skewness', 'curtosis', 'entropy']
target = 'class'


# splitting function
def split_train_test(df, ratio):  # a shuffle may be needed before the split
    mid = int(len(df) * ratio)  # split point
    train = df[:mid]
    test = df[mid:]
    return train, test


counter = 0

while counter != 5:
    print(f"test: {counter}")
    df = df.reindex(np.random.permutation(df.index))  # Shuffling
    train, test = split_train_test(df, 0.25)  # Split
    clf = tree.DecisionTreeClassifier()
    clf = clf.fit(train[features], train[target])
    res_pred = clf.predict(test[features])
    score = accuracy_score(test[target], res_pred)
    print(f"accuracy: {score}")
    treeObj = clf.tree_
    print(f"treeSize: {treeObj.node_count}")
    counter += 1

accuracy_list = []
treeSize_list = []

for train_size in range(30, 80, 10):

    counter = 0
    mean_accuracy = 0
    max_accuracy = -99999999999999.0
    min_accuracy = 99999999999999.0

    mean_tree_size = 0
    max_tree_size = -99999999999999.0
    min_tree_size = 99999999999999.0

    print(f"trainSize: {train_size}")
    while counter != 5:

        df = df.reindex(np.random.permutation(df.index))
        train, test = split_train_test(df, train_size / 100.0)
        clf = tree.DecisionTreeClassifier()
        clf = clf.fit(train[features], train[target])
        res_pred = clf.predict(test[features])

        score = accuracy_score(test[target], res_pred)

        mean_accuracy += score
        if max_accuracy < score:
            max_accuracy = score
        if min_accuracy > score:
            min_accuracy = score

        treeObj = clf.tree_

        mean_tree_size += treeObj.node_count
        if max_tree_size < treeObj.node_count:
            max_tree_size = treeObj.node_count
        if min_tree_size > treeObj.node_count:
            min_tree_size = treeObj.node_count

        counter += 1

    mean_accuracy = mean_accuracy / 5.0
    mean_tree_size = mean_tree_size / 5.0

    accuracy_list.append(mean_accuracy)

    treeSize_list.append(mean_tree_size)

    print(f"accuracy mean: {mean_accuracy}", end=" ")
    print(f"accuracy max: {max_accuracy}", end=" ")
    print(f"accuracy min: {min_accuracy}")

    print(f"tree size mean: {mean_tree_size}", end=" ")
    print(f"tree size max: {max_tree_size}", end=" ")
    print(f"tree size min: {min_tree_size}")

    print("----------------------------------------------------")

# clf = tree.DecisionTreeClassifier()
# clf = clf.fit(df[features], df[target])
# fig = plt.figure(figsize=(25, 20))
# treeplot = tree.plot_tree(clf)
# fig.savefig("decistion_tree.png")
plt.scatter([*range(30, 80, 10)], accuracy_list)
plt.xlabel("Training Size")
plt.ylabel("Accuracy")
plt.show()
plt.scatter([*range(30, 80, 10)], treeSize_list)
plt.xlabel("Training Size")
plt.ylabel("Tree Size")
plt.show()
