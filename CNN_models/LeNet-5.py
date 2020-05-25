import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D,Dense,MaxPooling2D,Flatten,Dropout,BatchNormalization,Activation
from tensorflow import keras
from tensorflow.keras.utils import plot_model
import os
from scipy import misc
import numpy as np
from tensorflow.keras.utils import to_categorical

class LeNet_5():
    def __init__(self):
        self.model=Sequential()
        self.model.add(Conv2D(filters=6, kernel_size=(5, 5),strides=(1,1),activation='relu', input_shape=(32, 32, 3)))
        self.model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        self.model.add(Conv2D(filters=6,kernel_size=(5,5),strides=(1,1),activation='relu'))
        self.model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        self.model.add(Flatten())
        self.model.add(Dense(units=500,activation='relu'))
        self.model.add(BatchNormalization(axis=-1))
        #self.model.add(Dropout(rate=0.6))
        self.model.add(Dense(units=500, activation='relu'))
        self.model.add(BatchNormalization(axis=-1))
        #self.model.add(Dropout(rate=0.6))
        self.model.add(Dense(units=10, activation='softmax'))
        plot_model(self.model,to_file=r'C:/Users/Zhiyan/Desktop/CNN_model/lenet-5.png',show_shapes=True)
        print(self.model.summary())

    def train(self,x,y):
        self.model.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
        self.model.fit(x,y,batch_size=24,epochs=100)

    def predict(self,x,y):
        loss,acc=self.model.evaluate(x,y,steps=1,verbose=0)
        print(loss,acc)

#load data
os.chdir(r'C:/Users/Zhiyan/Desktop/Sign-Language-Digits-Dataset/')
files=os.listdir('Dataset/')

def load_data():
    for file_idx,file in enumerate(files):
        cur_add='Dataset/'+file+'/'
        pics=os.listdir(cur_add)
        for idx,pic in enumerate(pics):
            # set the number of samples
            if idx==10: break
            if file_idx==0 and idx==0:
                picture=misc.imread(cur_add+pic)
                picture= misc.imresize(picture, size=(32,32, 3))
                picture=picture/255
                picture = picture[np.newaxis, :]
                Dataset=picture
                labels=[[idx]]
                continue
            picture=misc.imread(cur_add+pic)
            picture = misc.imresize(picture, size=(32,32, 3))
            picture = picture / 255
            picture=picture[np.newaxis,:]
            Dataset=np.concatenate((Dataset,picture))
            cur_label=[[int(file)]]
            labels=np.concatenate((labels,cur_label))
    #shuffle data
    shuffle=np.random.RandomState(seed=42).permutation(Dataset.shape[0])
    Dataset=Dataset[shuffle,:]
    labels=labels[shuffle,:]
    # one-hot encoding
    labels = to_categorical(labels)
    # split dataset to train,test set
    split_point=0.8
    m=Dataset.shape[0]
    train_x,test_x=Dataset[:int(split_point*m),:],Dataset[int((split_point)*m):,:]
    train_y,test_y=labels[:int(split_point*m),:],labels[int((split_point)*m):,:]
    print(train_x.shape)
    return train_x,train_y,test_x,test_y

def main():
    train_x,train_y,test_x,test_y=load_data()
    cnn=LeNet_5()
    cnn.train(train_x,train_y)
    cnn.predict(test_x,test_y)

if __name__=='__main__':
    main()
