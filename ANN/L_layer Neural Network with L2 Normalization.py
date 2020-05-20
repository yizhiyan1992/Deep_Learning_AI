import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
os.chdir('/Users/Zhiyan1992/Desktop/')

class L_layer_Neural_Network():
    def __init__(self):
        self.layer_list=[5]
        self.parameters={}
        self.caches={}

    def train_model(self,X,Y,iteration=5000,learning_rate=0.1,layer_list=None,lambd=0):
        if layer_list:
            self.layer_list=layer_list
        self.X=X
        self.Y=Y
        self.initialize_parameter()
        self.loss_val=[]
        for i in range(iteration):
            self.forward_propagation(self.X)
            loss_val=self.compute_loss_val(Y,self.y_prediction,lambd)
            self.compute_derivative(self.caches,self.y_prediction,self.parameters,self.Y)
            self.update_parameters(self.parameters,self.deri,learning_rate,lambd,Y.shape[1])
            self.loss_val.append(loss_val)
        #plt.plot(range(len(self.loss_val)),self.loss_val)
        #plt.show()
        return self.loss_val

    def initialize_parameter(self):
        '''
        :param no_feature: the number of features for training samples
        :param layer_list: a list to define how many neurons for each hidden layer
        :param y_shape: output dim
        :return: initial parameters of W and b (a dict)
        '''
        no_feature=self.X.shape[0]
        y_shape=self.Y.shape[0]
        np.random.seed(41)
        for i in range(len(self.layer_list)):
            if i==0:
                W=np.random.randn(self.layer_list[i],no_feature)
                b=np.zeros((self.layer_list[i],1))
            else:
                W = np.random.randn(self.layer_list[i], self.layer_list[i-1])
                b = np.zeros((self.layer_list[i], 1))
            self.parameters['W'+str(i+1)]=W
            self.parameters['b' + str(i + 1)] = b
        #last layer
        W = np.random.randn(y_shape, self.layer_list[-1])
        b = np.zeros((y_shape, 1))
        self.parameters['W' + str(len(self.layer_list) + 1)] = W
        self.parameters['b' + str(len(self.layer_list) + 1)] = b

        return

    def forward_propagation(self,X):
        '''
        :param X: input m training examples
        :param parameters: W and b
        :return: predicted result y_hat, and caches to store A for each layer (A[l],caches)
        formula:
        Z[l]=W[l]A[l-1]+b[l]
        A[l]=activation(Z[l])
        here, we use sigmoid func as activation function
        '''
        self.caches={}
        A=X
        self.caches['Z' + str(0)] = X
        self.caches['A' + str(0)] = X
        for i in range(len(self.parameters)//2):
            Z=np.dot(self.parameters['W'+str(i+1)],A)
            A=1/(1+np.exp(-Z))
            self.caches['Z'+str(i+1)]=Z
            self.caches['A'+str(i+1)]=A
        self.y_prediction=A
        return

    def compute_loss_val(self,Y,Y_prediction,lambd):
        '''
        :param Y:
        :param Y_prediction:
        :return: J(Y,Y_hat)
        use log-loss function to calculate total cost (note that y={0,1}, if not, it needs to be transformed into 0-1 vals)
        '''
        #compute regularizaion term
        L2_regularization=0
        for param in range(len(self.parameters)//2):
            L2_regularization+=np.sum(np.square(self.parameters['W'+str(param+1)]))
        m=Y.shape[1]
        #compute total loss
        loss=-(1/m)*(np.dot(Y,np.log(Y_prediction).T)+np.dot((1-Y),np.log(1-Y_prediction).T))+lambd/(2*m)*L2_regularization
        self.loss=loss[0][0]
        return self.loss

    def compute_derivative(self,caches,prediction,parameters,Y):
        m=prediction.shape[1]
        dZ=(prediction-Y)
        self.deri={}
        for i in reversed(range(len(parameters)//2)):
            dW=np.dot(dZ,caches['A'+str(i)].T)*(1/m)
            db=np.sum(dZ,axis=1,keepdims=True)*(1/m)
            self.deri['dW' + str(i + 1)] = dW
            self.deri['db' + str(i + 1)] = db
            dA_prev=np.dot(parameters['W'+str(i+1)].T,dZ)
            dZ_prev=dA_prev*(np.exp(-caches['Z'+str(i)])/((1+np.exp(-caches['Z'+str(i)]))*(1+np.exp(-caches['Z'+str(i)]))))
            dZ=dZ_prev
        return

    def update_parameters(self,parameters,deri,learning_rate,lambd,m):
        for i in range(len(parameters)//2):
            parameters['W'+str(i+1)]=(1-learning_rate*lambd/m)*parameters['W'+str(i+1)]-learning_rate*deri['dW'+str(i+1)]
            parameters['b' + str(i + 1)] = parameters['b' + str(i + 1)] - learning_rate *deri['db' + str(i + 1)]
        return

    def predict(self,X):
        A = X
        for i in range(len(self.parameters) // 2):
            Z = np.dot(self.parameters['W' + str(i + 1)], A)
            A = 1 / (1 + np.exp(-Z))
        res=np.where(A>=0.5,1,0)
        return res

    def score(self,Y,Y_predict):
        m=Y.shape[1]
        return np.sum(Y==Y_predict)/m

def plot_result(x,y,x3,y3,res_p):
    x1=x[0,:]
    x2=x[1,:]
    plt.pcolormesh(x3, y3, res_p,cmap='cool')
    plt.scatter(x1,x2,c=y[0])
    return

def main():
    '''
    preprocessing data:
    data shape should be [N,M], where N is the number of features, and M is the sample size
    '''
    train=pd.read_csv('bank-note/train.csv',header=None)
    test=pd.read_csv('bank-note/test.csv',header=None)
    X,Y=train.values[:,1:3],train.values[:,-1]
    X_test,Y_test=test.values[:,1:3],test.values[:,-1]
    Y=Y[:,np.newaxis]
    Y_test = Y_test[:, np.newaxis]
    X,Y=X.T,Y.T
    X_test,Y_test=X_test.T,Y_test.T

    # train model and predict samples
    plt.subplot()
    plt.subplot(2,2,1)
    lambd=[0,0.01,0.5]
    ANN_models={}
    for index,l in enumerate(lambd):
        ANN_models[index]=L_layer_Neural_Network()
        ANN_models[index].train_model(X,Y,iteration=10000,learning_rate=0.1,layer_list=[20,5],lambd=l)
        res = ANN_models[index].predict(X)
        print('prediction accuracy on training set: ', ANN_models[index].score(Y, res))
        res_test = ANN_models[index].predict(X_test)
        print('prediction accuracy on test set: ', ANN_models[index].score(Y_test, res_test))
        plt.plot(range(len(ANN_models[index].loss_val)), ANN_models[index].loss_val,label='lambda='+str(l))
    plt.legend()

    #plot
    for i in range(len(lambd)):
        plt.subplot(2, 2, i+2)
        xp = np.arange(-15, 15, 0.1)
        yp = np.arange(-15, 20, 0.1)
        xp, yp = np.meshgrid(xp, yp)
        x1,y1=xp,yp
        xp = xp.reshape(-1)
        yp = yp.reshape(-1)
        Xp = np.array(list(zip(xp, yp)))
        res_p = ANN_models[i].predict(Xp.T)
        res_p=res_p.reshape(x1.shape)
        plot_result(X, Y,x1,y1,res_p)
    plt.show()

if __name__=="__main__":
    main()
