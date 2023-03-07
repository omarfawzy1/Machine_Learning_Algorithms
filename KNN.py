import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


pd.options.mode.chained_assignment = None
df = pd.read_csv("BankNote_Authentication.csv")

features = ['variance', 'skewness', 'curtosis', 'entropy']
target = 'class'

df = df.reindex(np.random.permutation(df.index))


def split_train_test(dataframe, ratio):
    mid = int(len(dataframe) * ratio)  # split point
    train = dataframe[:mid]
    test = dataframe[mid:]
    train['distance'] = 0
    return train, test


train, test = split_train_test(df, 0.70)


# z = (x - mean) / std.
def normalize(feature):
    std = train[feature].std() * (1/100)
    mean = train[feature].mean()
    train[feature] = train[feature].apply(lambda x: (x - mean) / std)  # applying normalization function
    test[feature] = test[feature].apply(lambda x: (x - mean) / std)


def distance(dimensions):
    result = np.sqrt(np.sum((train[features].sub(dimensions, axis='columns')).pow(2), axis=1))
    return result


for feature in features:
    normalize(feature)


# Experiment

# print(distance([3.4566, 9.5228, -4.0112, -3.5944]))
# print(train.iloc[1:5])
# temp=train[features].sub([3.4566, 9.5228, -4.0112, -3.5944], axis='columns')
# temp=distance([3.4566, 9.5228, -4.0112, -3.5944])
# print(temp)

def predict(test, train, K):
    res = []
    temp = test[features].to_numpy().tolist()
    for i in range(len(temp)):
        dist = distance(temp[i])
        #train.loc[:, 'distance'] = dist
        train['distance'] = dist
        trueCount = 0
        falseCount = 0
        train = train.sort_values(by=['distance'])
        for j in range(K):
            if train['class'].iloc[j] == 1:
                trueCount += 1
            else:
                falseCount += 1

        if trueCount > falseCount:
            res.append(1)
        elif trueCount < falseCount:
            res.append(0)
        else:
            res.append(train['class'].iloc[0])
    return res


cases = range(1, 10)

for k in cases:
    print(f"k value : {k}")
    res = predict(test[features], train, k)
    y = test[target]
    correct = np.sum(res == y)
    print(f"Number of correctly classified instances : {correct} Total number of instances : {y.size}")
    print(f"Accuracy : {correct / y.size}")


# Double Test
#     knn = KNeighborsClassifier(n_neighbors=k)
#     knn.fit(train[features], train[target])
#     y_pred = knn.predict(test[features])
#     print(f"Actual Accuracy {metrics.accuracy_score(test[target], y_pred)}")


