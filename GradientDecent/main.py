import numpy as np

#using hit and trial method to decide iterations for optimisations

def gradientDecent(x,y):
    slope_current = intercept_current = 0
    iterations = 1000
    learning_rate = 0.01
    total_length = len(x)
    for i in range(iterations):
        y_predicted = slope_current * x + intercept_current
        cost = (1 / total_length) * sum([val ** 2 for val in (y - y_predicted)])
        partial_derivative_slope = -(2 / total_length) * sum(x * (y - y_predicted))
        partial_derivative_intercept = -(2 / total_length) * sum(y - y_predicted)
        slope_current = slope_current - learning_rate * partial_derivative_slope
        intercept_current = intercept_current - learning_rate * partial_derivative_intercept
        print("m {}, b {}, cost {} iteration {}".format(slope_current, intercept_current, cost, i))




x= np.array([1,2,3,4,5,6,7,10,8,9])
y =np.array([2,4,5,6,3,4,5,3,2,7])

gradientDecent(x,y)