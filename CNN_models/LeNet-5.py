# LeNet-5

import numpy as np
import os
import tensorflow as tf
from tensorflow.keras.utils import plot_model
import pydot
#from IPython.display import SVG
from tensorflow.keras import models
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import ZeroPadding2D
from tensorflow.keras.layers import BatchNormalization

import pandas as pd
from scipy import misc
print(tf.__version__)
os.chdir('/Users/Zhiyan1992/Documents/GitHub/Sign-Language-Digits-Dataset/')
files=os.listdir('Dataset/')
files.remove('.DS_Store')
print(files)


def load_data():
    for file_idx,file in enumerate(files):
        cur_add='Dataset/'+file+'/'
        pics=os.listdir(cur_add)
        if '.DS_Store' in pics:
            pics.remove('.DS_Store')

        for idx,pic in enumerate(pics):
            # set the number of samples
            if idx==20: break
            if file_idx==0 and idx==0:
                picture=misc.imread(cur_add+pic)
                picture= misc.imresize(picture, size=(100, 100, 3))
                picture=picture/255
                picture = picture[np.newaxis, :]
                Dataset=picture
                labels=[[idx]]
                continue
            picture=misc.imread(cur_add+pic)
            picture = misc.imresize(picture, size=(100, 100, 3))
            picture = picture / 255
            picture=picture[np.newaxis,:]
            Dataset=np.concatenate((Dataset,picture))
            cur_label=[[int(file)]]
            labels=np.concatenate((labels,cur_label))
    #shuffle data
    shuffle=np.random.permutation(Dataset.shape[0])
    Dataset=Dataset[shuffle,:]
    labels=labels[shuffle,:]
    labels=labels.astype('float32')
    # one-hot encoding
    labels=tf.one_hot(indices=labels[:,0],depth=10)
    # split dataset to train,test set
    split_point=0.8
    m=Dataset.shape[0]
    print(int(split_point*m),'!!!!')
    train_x,test_x=Dataset[:int(split_point*m),:],Dataset[int((split_point)*m):,:]
    train_y,test_y=labels[:int(split_point*m),:],labels[int((split_point)*m):,:]
    print(train_x.shape)
    return train_x,train_y,test_x,test_y


class LeNet_5():
    def __init__(self):
        self.model=models.Sequential()

    def create_model(self):

        self.model.add(Conv2D(filters=6, kernel_size=(5, 5), padding='valid', strides=(1, 1),input_shape=(100, 100, 3)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=16,kernel_size=(5,5),strides=(1,1)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(AveragePooling2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=32, kernel_size=(5, 5), strides=(1, 1)))
        self.model.add(Activation('relu'))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(AveragePooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=120))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=84))
        self.model.add(BatchNormalization(axis=-1))
        self.model.add(Activation('relu'))
        self.model.add(Dropout(rate=0.5))
        self.model.add(Dense(units=10))
        self.model.add(Activation('softmax'))

        self.model.summary()
        return

    def train_model(self,X_train,Y_train):
        X_train=tf.convert_to_tensor(X_train)
        Y_train=tf.convert_to_tensor(Y_train)
        self.model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['accuracy'])
        self.model.fit(X_train,Y_train,steps_per_epoch=4,epochs=30)
        self.model.save_weights(filepath='/Users/Zhiyan1992/Desktop/cnn/model_weights.h5')

    def predict(self,X,Y):
        print('prediction results:')
        loss,acc=self.model.evaluate(X,Y,steps=1,verbose=0)
        print(loss,acc)

def main():

    X_train,Y_train,X_test,Y_test=load_data()
    print(X_train.shape,Y_train.shape,X_test.shape,Y_test.shape)
    cnn=LeNet_5()
    cnn.create_model()
    cnn.train_model(X_train,Y_train)
    cnn.predict(X_test,Y_test)

    #cnn.model.load_weights(filepath='/Users/Zhiyan1992/Desktop/cnn/model_weights.h5')
    #loss,acc=cnn.model.evaluate(X_test,Y_test,steps=1)

if __name__=='__main__':
    main()
