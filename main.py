import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression


# gradient descent function from scratches
def gradient_descent(x, y):
    # initialize th0 and th1 as 0
    th0 = th1 = 0
    # set number of iterations
    iterations = 100000
    # set number of examples
    n = len(x)
    # set learning rate
    alpha = 0.0002
    # create df for training data
    training = pd.DataFrame(columns=['th0', 'th1', 'cost', 'iteration'])

    # iterate prediction
    for i in range(iterations):
        # predict y
        y_pred = th0 + th1 * x
        # compute cost
        cost = (1/n) * sum([val**2 for val in (y-y_pred)])
        # compute derivative of th0
        th0d = -(2/n) * sum(y - y_pred)
        # compute derivative of th1
        th1d = -(2/n) * sum(x * (y - y_pred))
        # update th0
        th0 = th0 - alpha * th0d
        # update th1
        th1 = th1 - alpha * th1d
        # print at each iteration th0, th1, and cost
        print("th0 {}, th1 {}, cost {} iteration {}".format(th0,th1,cost, i))
        training = training.append({'th0':th0, 'th1':th1, 'cost':cost, 'iteration':i},
                                       ignore_index=True)

    return training


# sklean prediction function
def predict_with_sklean(df):
    # create lin reg obj
    r = LinearRegression()
    # fit lin reg
    r.fit(df[['math']],df.cs)
    return r.intercept_, r.coef_


# load df
df = pd.read_csv('exams.csv')

# with GRADIENT DESCENT
# assign x an y
x = np.array(df.math)
y = np.array(df.cs)

# store training of gradient descent funct
training = gradient_descent(x,y)

# plot cost and iteration
plt.scatter(training.iteration, training.cost, color='blue', marker='o')
plt.xlabel('iteration')
plt.ylabel('cost')
plt.show()

print('Using gradient descent function: Intercept {} Coef {} '.format(training.th0.iloc[-1], training.th1.iloc[-1]))

# with SKLEAN
th0_sklearn, th1_sklearn = predict_with_sklean(df)
print("Using sklearn: Intercept {} Coef {}".format(th0_sklearn,th1_sklearn))

