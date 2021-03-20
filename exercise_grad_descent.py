import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import math

def predict_using_sklearn():
    df=pd.read_csv("test_scores.csv")
    r=LinearRegression()
    r.fit(df[['math']],df.cs)
    return r.coef_, r.intercept_

def gradient_descent(x,y):
    m_curr=b_curr=0
    iterations=1000000
    learning_rate=0.0002
    n=len(x)
    prev_cost=0
    for i in range(iterations):
        y_predicted=m_curr * x + b_curr
        cost=(1/2) * sum([val**2 for val in (y-y_predicted)])
        md=-(2/n) * sum(x*(y-y_predicted))
        bd=-(2/n) * sum(y - y_predicted)
        m_curr=m_curr-learning_rate * md
        b_curr=b_curr-learning_rate * bd
        if math.isclose(cost,prev_cost,rel_tol=1e-20):
            break
            cost=prev_cost
        print('m {},b {},cost {}, iteration {}'.format(m_curr,b_curr,cost,i))

    return m_curr, b_curr


if __name__=="__main__":
    df = pd.read_csv('test_scores.csv')
    x = np.array(df.math)
    y = np.array(df.cs)

    m, b = gradient_descent(x,y)
    print("using Gradient Descent function: coefficient {}, intercept {}" .format(m,b))

    m_sklearn, b_sklearn = predict_using_sklearn()
    print("Using sklearn: coefficient {}, intercept {}".format(m_sklearn, b_sklearn))

