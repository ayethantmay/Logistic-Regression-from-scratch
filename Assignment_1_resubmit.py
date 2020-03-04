import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from matplotlib import pyplot as plt
from matplotlib import interactive

df = pd.read_csv('IRIS.csv')
df['species'] = pd.Categorical(df['species'])
df_temporary = pd.get_dummies(df['species'])
df = pd.concat([df, df_temporary],axis=1)

X = df.drop(df.columns[[4,5,6,7]], axis=1)
Y_Setosa = df.drop(df.columns[[0,1,2,3,4,6,7]],axis=1)
Y_Versicolor = df.drop(df.columns[[0,1,2,3,4,5,7]],axis=1)
Y_Virginica = df.drop(df.columns[[0,1,2,3,4,5,6]],axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)
X_train_Setosa, X_test_Setosa, y_train_Setosa, y_test_Setosa = train_test_split(X, Y_Setosa, test_size=0.2, random_state=42)
X_train_Versicolor, X_test_Versicolor, y_train_Versicolor, y_test_Versicolor = train_test_split(X, Y_Versicolor, test_size=0.2, random_state=42)
X_train_Virginica, X_test_Virginica, y_train_Virginica, y_test_Virginica = train_test_split(X, Y_Virginica, test_size=0.2, random_state=42)

X_train_Setosa = np.array(X_train_Setosa)
X_test_Setosa = np.array(X_test_Setosa)
y_train_Setosa = np.array(y_train_Setosa)
y_test_Setosa = np.array(y_test_Setosa)

X_train_Versicolor = np.array(X_train_Versicolor)
X_test_Versicolor = np.array(X_test_Versicolor)
y_train_Versicolor = np.array(y_train_Versicolor)
y_test_Versicolor = np.array(y_test_Versicolor)

X_train_Virginica = np.array(X_train_Virginica)
X_test_Virginica = np.array(X_test_Virginica)
y_train_Virginica = np.array(y_train_Virginica)
y_test_Virginica = np.array(y_test_Virginica)


def weightInitialization(n_features):
    w = np.zeros((1,n_features))
    b = 0
    return w,b   
    
def Logistic_Regression(result):
    final_result = 1/(1+np.exp(-result))
    return final_result

def Model_optimization(w, b, X, Y, penalty):
    m = X.shape[0]
    
    #Prediction
    final_result = Logistic_Regression(np.dot(w,X.T)+b)
    Y_T = Y.T
    result = 0
    result1 = 0

    #loss with Ridge Regularization function 
    for j in range(n_features):
        result += w[:,j]**2
    regularization = penalty * result

    cost = (-1/m)*(np.sum(( Y_T*np.log(final_result)) + ((1-Y_T)*(np.log(1-final_result))))) + regularization 

    #Gradient with Ridge Regularization calculation 
    for k in range(n_features):
        result1 += 2 * penalty * abs(w[:,k])

    dw = (1/m)*(np.dot(X.T, (final_result-Y.T).T)) + result1
    db = (1/m)*(np.sum(final_result-Y.T))   
    grads = {"dw": dw, "db": db} 
    return grads, cost

def Model_prediction(w, b, X, Y, penalty, learning_rate, no_iterations):
    cost_history = []
    iterations = []
    for i in range(no_iterations):
        grads, cost = Model_optimization(w,b,X,Y,penalty)
        dw = grads["dw"]
        db = grads["db"]
        #weight update
        #Gradient Descent with Ridge Regularization calculation 
        w = w - (learning_rate * (dw.T))
        b = b - (learning_rate * db)
        
        if (i % 100 == 0):
            iterations.append(i)
            cost_history.append(cost)
    
    #final parameters
    coeff = {"w": w, "b": b}
    gradient = {"dw": dw, "db": db}
    
    return coeff, gradient, cost_history, iterations


def Model_Prediction_sgd(w, b, X, Y, penalty, learning_rate, no_iterations):
    cost_history = []
    iterations = []
    m = X.shape[0]
    #batch_size = 2,10,50
    batch_size = 10
    cost = 0
    #n_batch = int(m/batch_size)

    for i in range(no_iterations):
        # Stochastic Gradient Descent with Ridge Regularization calculation 
        indices = np.random.permutation(m)
        X = X[indices]
        Y = Y[indices]
        for j in range(0,m,batch_size):
            X_j = X[j:j+batch_size]
            Y_j = Y[j:j+batch_size]
            grads, cost = Model_optimization(w,b,X_j,Y_j,penalty)
            cost += cost

            dw = grads["dw"]
            db = grads["db"]
            #weight update
            w = w - (learning_rate * (dw.T))
            b = b - (learning_rate * db)
        
        if (i % 100 == 0):
            iterations.append(i)
            cost_history.append(cost)
    
    #final parameters
    coeff_sgd = {"w": w, "b": b}
    gradient_sgd = {"dw": dw, "db": db}
    
    return coeff_sgd, gradient_sgd, cost_history, iterations



def Finalize_Predicted_Result(final_pred, m):
    y_pred = np.zeros((1,m))
    for i in range(final_pred.shape[1]):
        if final_pred[0][i] > 0.5:
            y_pred[0][i] = 1
    return y_pred


n_features = X_train_Setosa.shape[1]

l = [i for i in np.arange(0.01,0.201,0.01)]


### Main for Setosa ###
cost_history_Setosa = []
iterations_Setosa = []
weight_1_setosa = []
weight_2_setosa = []
weight_3_setosa = []
weight_4_setosa = []
f1 = plt.figure(1)

for i in l:
    w_Setosa, b_Setosa = weightInitialization(n_features)
    coeff_Setosa, gradient_Setosa, cost_history_Setosa, iterations_Setosa = Model_prediction(w_Setosa, b_Setosa, X_train_Setosa, y_train_Setosa, penalty = i, learning_rate=0.00001,no_iterations=2000)
    w_Setosa = coeff_Setosa["w"]
    b_Setosa = coeff_Setosa["b"]
  
    weight_1_setosa.append(w_Setosa[0,0])
    weight_2_setosa.append(w_Setosa[0,1])
    weight_3_setosa.append(w_Setosa[0,2])
    weight_4_setosa.append(w_Setosa[0,3])
    

    final_train_pred_Setosa = Logistic_Regression(np.dot(w_Setosa,X_train_Setosa.T)+b_Setosa)
    final_test_pred_Setosa = Logistic_Regression(np.dot(w_Setosa,X_test_Setosa.T)+b_Setosa)
    plt.plot(l,cost_history_Setosa,label=str(i))
plt.xlabel('Regularization rates')
plt.ylabel('Losses')
plt.title("Regularization rates Vs Losses (Setosa)")
plt.legend(prop={'size': 5})
interactive(True)
plt.show()
f2 = plt.figure(2)
plt.plot(l,weight_1_setosa,label='weight 1')
plt.plot(l,weight_2_setosa,label='weight 2')
plt.plot(l,weight_3_setosa,label='weight 3')
plt.plot(l,weight_4_setosa,label='weight 4')
plt.xlabel('Regularization rates')
plt.ylabel('Weights')
plt.title("Regularization rates Vs Weights (Setosa)")
plt.legend()
interactive(True)
plt.show()   

    #plt.plot(l,cost_history_Setosa, label=str(i))
#plt.plot(l,cost_history_Setosa,color='r',label=str())


### Main for Versicolor ###
cost_history_Versicolor = []
iterations_Versicolor = []
weight_1_versicolor = []
weight_2_versicolor = []
weight_3_versicolor = []
weight_4_versicolor = []
f3 = plt.figure(3)

for i in l:

    w_Versicolor, b_Versicolor = weightInitialization(n_features)
    coeff_Versicolor, gradient_Versicolor, cost_history_Versicolor, iterations_Versicolor = Model_prediction(w_Versicolor, b_Versicolor, X_train_Versicolor, y_train_Versicolor, penalty = i, learning_rate=0.00001,no_iterations=2000)
    w_Versicolor = coeff_Versicolor["w"]
    b_Versicolor = coeff_Versicolor["b"]
    weight_1_versicolor.append(w_Versicolor[0,0])
    weight_2_versicolor.append(w_Versicolor[0,1])
    weight_3_versicolor.append(w_Versicolor[0,2])
    weight_4_versicolor.append(w_Versicolor[0,3])

    final_train_pred_Versicolor = Logistic_Regression(np.dot(w_Versicolor,X_train_Versicolor.T)+b_Versicolor)
    final_test_pred_Versicolor = Logistic_Regression(np.dot(w_Versicolor,X_test_Versicolor.T)+b_Versicolor)
    plt.plot(l,cost_history_Versicolor,label=str(i))
plt.xlabel('Regularization rates')
plt.ylabel('Losses')
plt.title("Regularization rates Vs Losses (Versicolor)")
plt.legend(prop={'size': 5})
interactive(True)
plt.show()
f4 = plt.figure(4)
plt.plot(l,weight_1_versicolor,label='weight 1')
plt.plot(l,weight_2_versicolor,label='weight 2')
plt.plot(l,weight_3_versicolor,label='weight 3')
plt.plot(l,weight_4_versicolor,label='weight 4')
plt.xlabel('Regularization rates')
plt.ylabel('Weights')
plt.title("Regularization rates Vs Weights (Versicolor)")
plt.legend()
interactive(True)
plt.show()   
#plt.plot(l,cost_history_Versicolor,color='g',label='Versicolor')


### Main for Virginica ###
cost_history_Virginica = []
iterations_Virginica = []
weight_1_virginica = []
weight_2_virginica = []
weight_3_virginica = []
weight_4_virginica = []
f5 = plt.figure(5)
for i in l:

    w_Virginica, b_Virginica = weightInitialization(n_features)
    coeff_Virginica, gradient_Virginica, cost_history_Virginica, iterations_Virginica = Model_prediction(w_Virginica, b_Virginica, X_train_Virginica, y_train_Virginica, penalty = i, learning_rate=0.00001,no_iterations=2000)
    w_Virginica = coeff_Virginica["w"]
    b_Virginica = coeff_Virginica["b"]
    weight_1_virginica.append(w_Virginica[0,0])
    weight_2_virginica.append(w_Virginica[0,1])
    weight_3_virginica.append(w_Virginica[0,2])
    weight_4_virginica.append(w_Virginica[0,3])

    final_train_pred_Virginica = Logistic_Regression(np.dot(w_Virginica,X_train_Virginica.T)+b_Virginica)
    final_test_pred_Virginica = Logistic_Regression(np.dot(w_Virginica,X_test_Virginica.T)+b_Virginica)
    plt.plot(l,cost_history_Virginica,label=str(i))
plt.xlabel('Regularization rates')
plt.ylabel('Losses')
plt.title("Regularization rates Vs Losses (Virginica)")
plt.legend(prop={'size': 5})
interactive(True)
plt.show()
f6 = plt.figure(6)
plt.plot(l,weight_1_virginica,label='weight 1')
plt.plot(l,weight_2_virginica,label='weight 2')
plt.plot(l,weight_3_virginica,label='weight 3')
plt.plot(l,weight_4_virginica,label='weight 4')
plt.xlabel('Regularization rates')
plt.ylabel('Weights')
plt.title("Regularization rates Vs Weights (Virginica)")
plt.legend()
interactive(True)
plt.show()   
#plt.plot(l,cost_history_Versicolor,color='g',label='Versicolor')


#------------------------------------------------


# Main for SGD (Setosa)
cost_history_sgd_Setosa = []
iterations_sgd_Setosa = []
weight_1_setosa_sgd = []
weight_2_setosa_sgd = []
weight_3_setosa_sgd = []
weight_4_setosa_sgd = []
f7 = plt.figure(7)
for i in l:

    w_sgd_Setosa, b_sgd_Setosa = weightInitialization(n_features)
    coeff_sgd_Setosa, gradient_sgd_Setosa, cost_history_sgd_Setosa, iterations_sgd_Setosa = Model_Prediction_sgd(w_sgd_Setosa, b_sgd_Setosa, X_train_Setosa, y_train_Setosa, penalty = i, learning_rate=0.00001,no_iterations=2000)
    #Final prediction
    w_sgd_Setosa = coeff_sgd_Setosa["w"]
    b_sgd_Setosa = coeff_sgd_Setosa["b"]
    weight_1_setosa_sgd.append(w_sgd_Setosa[0,0])
    weight_2_setosa_sgd.append(w_sgd_Setosa[0,1])
    weight_3_setosa_sgd.append(w_sgd_Setosa[0,2])
    weight_4_setosa_sgd.append(w_sgd_Setosa[0,3])

    final_train_pred_sgd_Setosa = Logistic_Regression(np.dot(w_sgd_Setosa,X_train_Setosa.T)+b_sgd_Setosa)
    final_test_pred_sgd_Setosa = Logistic_Regression(np.dot(w_sgd_Setosa,X_test_Setosa.T)+b_sgd_Setosa)
    plt.plot(l,cost_history_sgd_Setosa,label=str(i))
plt.xlabel('Regularization rates')
plt.ylabel('Losses')
plt.title("Regularization rates Vs Losses (Setosa SGD)")
plt.legend(prop={'size': 5})
interactive(True)
plt.show()
f8 = plt.figure(8)
plt.plot(l,weight_1_setosa_sgd,label='weight 1')
plt.plot(l,weight_2_setosa_sgd,label='weight 2')
plt.plot(l,weight_3_setosa_sgd,label='weight 3')
plt.plot(l,weight_4_setosa_sgd,label='weight 4')
plt.xlabel('Regularization rates')
plt.ylabel('Weights')
plt.title("Regularization rates Vs Weights (Setosa SGD)")
plt.legend()
interactive(True)
plt.show()   
   


# Main for SGD (Versicolor)
cost_history_sgd_Versicolor = []
iterations_sgd_Versicolor = []
weight_1_versicolor_sgd = []
weight_2_versicolor_sgd = []
weight_3_versicolor_sgd = []
weight_4_versicolor_sgd = []
f9 = plt.figure(9)
for i in l:

    w_sgd_Versicolor, b_sgd_Versicolor = weightInitialization(n_features)
    coeff_sgd_Versicolor, gradient_sgd_Versicolor, cost_history_sgd_Versicolor, iterations_sgd_Versicolor = Model_Prediction_sgd(w_sgd_Versicolor, b_sgd_Versicolor, X_train_Versicolor, y_train_Versicolor, penalty = i, learning_rate=0.00001,no_iterations=2000)
    #Final prediction
    w_sgd_Versicolor = coeff_sgd_Versicolor["w"]
    b_sgd_Versicolor = coeff_sgd_Versicolor["b"]
    weight_1_versicolor_sgd.append(w_sgd_Versicolor[0,0])
    weight_2_versicolor_sgd.append(w_sgd_Versicolor[0,1])
    weight_3_versicolor_sgd.append(w_sgd_Versicolor[0,2])
    weight_4_versicolor_sgd.append(w_sgd_Versicolor[0,3])

    final_train_pred_sgd_Versicolor = Logistic_Regression(np.dot(w_sgd_Versicolor,X_train_Versicolor.T)+b_sgd_Versicolor)
    final_test_pred_sgd_Versicolor = Logistic_Regression(np.dot(w_sgd_Versicolor,X_test_Versicolor.T)+b_sgd_Versicolor)
    plt.plot(l,cost_history_sgd_Versicolor,label=str(i))
plt.xlabel('Regularization rates')
plt.ylabel('Losses')
plt.title("Regularization rates Vs Losses (Versicolor SGD)")
plt.legend(prop={'size': 5})
interactive(True)
plt.show()
f10 = plt.figure(10)
plt.plot(l,weight_1_versicolor_sgd,label='weight 1')
plt.plot(l,weight_2_versicolor_sgd,label='weight 2')
plt.plot(l,weight_3_versicolor_sgd,label='weight 3')
plt.plot(l,weight_4_versicolor_sgd,label='weight 4')
plt.xlabel('Regularization rates')
plt.ylabel('Weights')
plt.title("Regularization rates Vs Weights (Versicolor SGD)")
plt.legend()
interactive(True)
plt.show() 


# Main for SGD (Virginica)
cost_history_sgd_Virginica = []
iterations_sgd_Virginica = []
weight_1_virginica_sgd = []
weight_2_virginica_sgd = []
weight_3_virginica_sgd = []
weight_4_virginica_sgd = []
f11 = plt.figure(11)
for i in l:

    w_sgd_Virginica, b_sgd_Virginica = weightInitialization(n_features)
    coeff_sgd_Virginica, gradient_sgd_Virginica, cost_history_sgd_Virginica, iterations_sgd_Virginica = Model_Prediction_sgd(w_sgd_Virginica, b_sgd_Virginica, X_train_Virginica, y_train_Virginica, penalty = i, learning_rate=0.00001,no_iterations=2000)
    #Final prediction
    w_sgd_Virginica = coeff_sgd_Virginica["w"]
    b_sgd_Virginica = coeff_sgd_Virginica["b"]
    weight_1_virginica_sgd.append(w_sgd_Virginica[0,0])
    weight_2_virginica_sgd.append(w_sgd_Virginica[0,1])
    weight_3_virginica_sgd.append(w_sgd_Virginica[0,2])
    weight_4_virginica_sgd.append(w_sgd_Virginica[0,3])

    final_train_pred_sgd_Virginica = Logistic_Regression(np.dot(w_sgd_Virginica,X_train_Virginica.T)+b_sgd_Virginica)
    final_test_pred_sgd_Virginica = Logistic_Regression(np.dot(w_sgd_Virginica,X_test_Virginica.T)+b_sgd_Virginica)
    plt.plot(l,cost_history_sgd_Virginica,label=str(i))
plt.xlabel('Regularization rates')
plt.ylabel('Losses')
plt.title("Regularization rates Vs Losses (Virginica SGD)")
plt.legend(prop={'size': 5})
interactive(True)
plt.show()
f12 = plt.figure(12)
plt.plot(l,weight_1_virginica_sgd,label='weight 1')
plt.plot(l,weight_2_virginica_sgd,label='weight 2')
plt.plot(l,weight_3_virginica_sgd,label='weight 3')
plt.plot(l,weight_4_virginica_sgd,label='weight 4')
plt.xlabel('Regularization rates')
plt.ylabel('Weights')
plt.title("Regularization rates Vs Weights (Virginica SGD)")
plt.legend()
interactive(True)
plt.show() 

f13 = plt.figure(13)
plt.plot(iterations_Setosa,cost_history_Setosa,color = 'r', label ="Setosa")
plt.plot(iterations_Versicolor,cost_history_Versicolor,color = 'g', label ="Versicolor")
plt.plot(iterations_Virginica,cost_history_Virginica,color = 'b', label ="Virginica")
plt.ylabel('Training Loss')
plt.xlabel('Training Iteration')
plt.title("Gradient Descent Optimization")
plt.legend()
interactive(True)
plt.show()

f14 = plt.figure(14)
plt.plot(iterations_sgd_Setosa,cost_history_sgd_Setosa,color = 'r', label ="Setosa")
plt.plot(iterations_sgd_Versicolor,cost_history_sgd_Versicolor,color = 'g', label ="Versicolor")
plt.plot(iterations_sgd_Virginica,cost_history_sgd_Virginica,color = 'b', label ="Virginica")
plt.ylabel('Training Loss')
plt.xlabel('Training Iteration')
plt.title("Stochastic Gradient Descent Optimization")
plt.legend()
interactive(False)
plt.show()