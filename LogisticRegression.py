import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Logistic Regression
df = pd.read_csv('customer_data.csv')
print("Logistic Regression started")

# normalizing function
# z = (x – min) / (max – min).
def normalize(df, feature):
    min = df[feature].min()
    max = df[feature].max()
    return df[feature].apply(lambda x: (x - min) / (max - min))  # applying normalization function


# splitting function
def split_train_test(df, ratio):  # a shuffle may be needed before the split
    mid = (int)(len(df) * ratio)  # split point
    train = df[:mid]
    test = df[mid:]
    return train, test


def hypothesis(x, t):
    tx = np.negative(np.sum(np.multiply(x, t), axis=1))
    return np.divide(1, np.add(1, np.exp(tx)))


features = ['age', 'salary']
target = 'purchased'

for feature in features:
    df[feature] = normalize(df, feature)
df = df.reindex(np.random.permutation(df.index))
train, test = split_train_test(df, 0.8)

theta = np.array([1.0, 1.0, 1.0])
old_theta = theta.copy()
alpha = 0.01
y = train[target].to_numpy()

x = train[features].to_numpy()
x = np.c_[np.ones(x.shape[0]), x]


def accuarcy(x, y):
    h = hypothesis(x, theta)
    h = np.round(h)
    correct = np.sum(h == y)
    return correct / y.size


for i in range(1000):
    h = hypothesis(x, theta)
    sum_error = np.subtract(y, h)
    for j in range(3):
        theta[j] += alpha * np.multiply(sum_error, x[:, j]).sum()
    #ac = accuarcy(x, y)
    # print(f"{ac} \t\t\t {sum_error.sum()}")

x = test[features].to_numpy()
x = np.c_[np.ones(x.shape[0]), x]
print(f"The accuracy on the test set is {accuarcy(x, test[target].to_numpy())}")
print(f"Theta {theta}")
