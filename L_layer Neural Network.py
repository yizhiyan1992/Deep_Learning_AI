import pandas as pd
import numpy as np
import math
import os
os.chdir('/Users/Zhiyan1992/Desktop/')

class L_layer_Neural_Network():
    def __init__(self,layer_list=[5]):
        self.layer_list=layer_list
        self.parameters={}
        self.caches={}

    def train_model(self,X,Y):
        self.X=X
        self.Y=Y
        self.initialize_parameter()
        self.forward_propagation(self.X)
        self.compute_loss_val(Y,self.y_prediction)
        return

    def initialize_parameter(self):
        '''
        :param no_feature: the number of features for training samples
        :param layer_list: a list to define how many neurons for each hidden layer
        :param y_shape: output dim
        :return: initial parameters of W and b (a dict)
        '''
        no_feature=self.X.shape[0]
        y_shape=self.Y.shape[0]
        np.random.seed(42)
        for i in range(len(self.layer_list)):
            if i==0:
                W=np.random.randn(self.layer_list[i],no_feature)*0.01
                b=np.zeros((self.layer_list[i],1))
            else:
                W = np.random.randn(self.layer_list[i], self.layer_list[i-1]) * 0.01
                b = np.zeros((self.layer_list[i], 1))
            self.parameters['W'+str(i+1)]=W
            self.parameters['b' + str(i + 1)] = b
        #last layer
        W = np.random.randn(y_shape, self.layer_list[-1]) * 0.01
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
        for i in range(len(self.parameters)//2):
            Z=np.dot(self.parameters['W'+str(i+1)],A)
            A=1/(1+np.exp(-Z))
            self.caches['Z'+str(i+1)]=Z
            self.caches['A'+str(i+1)]=A
        self.y_prediction=A
        return

    def compute_loss_val(self,Y,Y_prediction):
        '''
        :param Y:
        :param Y_prediction:
        :return: J(Y,Y_hat)
        use log-loss function to calculate total cost (note that y={0,1}, if not, it needs to be transformed into 0-1 vals)
        '''
        m=Y.shape[1]
        loss=-(1/m)*(np.dot(Y,np.log(Y_prediction).T)+np.dot((1-Y),np.log(1-Y_prediction).T))
        self.loss=loss[0][0]
        print(self.loss)
        return

def main():
    train=pd.read_csv('bank-note/train.csv',header=None)
    X,Y=train.values[:,:-1],train.values[:,-1]
    Y=Y[:,np.newaxis]
    X,Y=X.T,Y.T
    print(X.shape,Y.shape)
    # row represents features and column represent sample size
    ANN=L_layer_Neural_Network()
    ANN.train_model(X,Y)


if __name__=="__main__":
    main()
