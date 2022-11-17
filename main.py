import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
data = pd.read_excel(r'C:\Users\chamr\Downloads\Assignment #2\Data for assignment #2.xlsx', sheet_name='Sheet2')


X_training = [[], [], [], []]
X_testing = [[], [], [], []]
Y_training = []
Y_testing = []


for row in range(len(data['X1'])):

    if (row+1) % 5 == 0:
        X_testing[0].append(1)
        X_testing[1].append(data['X1'][row])
        X_testing[2].append(data['X2'][row])
        X_testing[3].append(data['X3'][row])
        Y_testing.append(data['Y'][row])
    else:
        X_training[0].append(1)
        X_training[1].append(data['X1'][row])
        X_training[2].append(data['X2'][row])
        X_training[3].append(data['X3'][row])
        Y_training.append(data['Y'][row])


m = len(Y_training)
theta = [0, 0, 0, 0]

alpha = 1
lamda = 1
C = 0.01


def normalize(x):
    for column in range(1, len(x)):
        for i in range(m):
            x[column][i] = (x[column][i] - min(x[column]))/(max(x[column])-min(x[column]))


def h_theta_xi(i, x, thet):
    theta_t_xi = thet[0] + thet[1]*x[1][i] + thet[2]*x[2][i] + thet[3]*x[3][i]
    # To prevent overflow
    if theta_t_xi > 500:
        theta_t_xi = 500
    elif theta_t_xi < -500:
        theta_t_xi = -500
    return 1 / (1 + np.exp(-theta_t_xi))


def meansquarerror(x, thet, y):
    error = 0
    max_error = 0
    for i in range(len(y)):
        error = error + (h_theta_xi(i, x, thet) - y[i])*(h_theta_xi(i, x, thet) - y[i])
        if abs(h_theta_xi(i, x, thet) - y[i]) > max_error:
            max_error = abs(h_theta_xi(i, x, thet) - y[i])
    return np.sqrt(error/len(y)), max_error


def theta_update(column, x, y, thet):
    sum_of_diff = 0
    for i in range(len(y)):
        sum_of_diff = sum_of_diff + (h_theta_xi(i, x, thet)-y[i])*x[column][i]
    derivative_of_costfunc = sum_of_diff/len(y) + lamda*abs(thet[column])/len(y)
    return thet[column] - alpha*derivative_of_costfunc


normalize(X_training)
theta_new = [0, 0, 0, 0]
theta_older = [0, 0, 0, 0]
training_error_array = []
testing_error_array = []
max_testing_error_array =[]
while 1:
    for j in range(4):
        theta_new[j] = theta_update(j, X_training, Y_training, theta)
    diff_old = 0
    diff_new = 0

    for j in range(4):
        if diff_new < abs(theta_new[j]-theta[j]):
            diff_new = abs(theta_new[j]-theta[j])
        if diff_old < abs(theta[j]-theta_older[j]):
            diff_old = abs(theta[j]-theta_older[j])
    for j in range(4):
        theta_older[j] = theta[j]
        theta[j] = theta_new[j]
    training_error = meansquarerror(X_training, theta_older, Y_training)[0]
    training_error_array.append(training_error)
    testing_error, max_testing_error = meansquarerror(X_testing, theta_older, Y_testing)
    testing_error_array.append(testing_error)
    max_testing_error_array.append(max_testing_error)
    if diff_new <= C and diff_old <= C:
        break

Y = []
errors = 0
for row in range(len(Y_testing)):
    if h_theta_xi(row, X_testing, theta) <= 0.5:
        Y.append(0)
    else:
        Y.append(1)
    if Y[row] != Y_testing[row]:
        errors += 1

print(f"{errors} errors out of {len(Y)}")
print(f"Y_modeled = {Y}")
print(f"Y_testing = {Y_testing}")
print(f"theta = {theta}")
print(f"Average training error = {training_error}, length ={len(training_error_array)}")
print(f"Average testing error = {testing_error}")
print(f"Max testing error = {max_testing_error}")
plt.title(f"Convergence History at Lambda = {lamda}")
plt.plot(training_error_array, ".")
plt.plot(testing_error_array, ".r")
plt.plot(max_testing_error_array, ".g")
plt.legend(["Average Training Error","Average Testing Error","Max Testing Error"])
plt.show()

