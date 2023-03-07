import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Liner Regression
df = pd.read_csv('car_data.csv')
print("Liner Regression started")

Y = 'price'
V = df
features = ['curbweight', 'enginesize', 'horsepower', 'highwaympg']

for X in features:
    plt.scatter(V[X], V[Y])
    plt.xlabel(X)
    plt.ylabel(Y)
    plt.show()


# splitting function
def split_train_test(df, ratio):  # a shuffle may be needed before the split
    mid = (int)(len(df) * ratio)  # split point
    train = df[:mid]
    test = df[mid:]
    return train, test


# normalizing function
# z = (x – min) / (max – min).
def normalize(df, feature):
    min = df[feature].min()
    max = df[feature].max()
    return df[feature].apply(lambda x: (x - min) / (max - min))  # applying normalization function


# Shuffling the data
df = df.reindex(np.random.permutation(df.index))

features = ['curbweight', 'enginesize', 'horsepower', 'highwaympg']
target = ['price']
# Features selected for the regression

for feature in features:
    df[feature] = normalize(df, feature)
df[features].describe()  # for debugging

df = df[features + target]  # dropping the other columns
train, test = split_train_test(df, 0.60)  # splitting data into train and test


def hypothesis(x, theta):
    return np.sum(np.multiply(x, theta), axis=1)


def mean_square_error(x, theta, y):
    h = hypothesis(x, theta)
    return np.mean(np.square(np.subtract(h, y)))


total_iterations = []
total_mse = []


def drawline(iter, mse):
    if iter < 2:
        return
    if iter % 2 == 0:
        total_mse.append(mse)
        total_iterations.append(iter)
    if iter % 199 == 0:
        plt.plot(total_iterations, total_mse)
        plt.show()


alpha = 0.00008
counter = 0
m = train.shape[0]  # number of rows
theta = np.array([1.0, 1.0, 1.0, 1.0, 1.0])
old_theta = theta.copy()
y = train[target].to_numpy()
x = train[features].to_numpy()
x = np.c_[np.ones(x.shape[0]), x]

while counter != 200:
    h = np.array([hypothesis(x, theta)]).transpose()
    sum_error = np.subtract(h, y)
    for j in range(5):
        theta[j] -= (alpha / m) * (np.multiply(sum_error, x[:, j]).sum())
    counter += 1
    drawline(counter, mean_square_error(x, theta, y))

x = test[features].to_numpy()
x = np.c_[np.ones(x.shape[0]), x]
y = test[target].to_numpy()

print(f"Mean square error {mean_square_error(x, theta, y)}")
print(f"Theta = {theta}")
