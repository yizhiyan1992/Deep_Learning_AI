from tensorflow.compat.v1 import enable_eager_execution
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Dropout,Input,Activation
from tensorflow.keras.utils import to_categorical
import matplotlib.pyplot as plt
import os
import tensorflow as tf
import numpy as np
from scipy import misc
print(tf.__version__)
os.chdir(r'C:/Users/Zhiyan/Desktop/Sign-Language-Digits-Dataset/')
files=os.listdir('Dataset/')

print(files)
def load_data():
    for file_idx,file in enumerate(files):
        cur_add='Dataset/'+file+'/'
        pics=os.listdir(cur_add)
        for idx,pic in enumerate(pics):
            # set the number of samples
            #if idx==50: break
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
    shuffle=np.random.RandomState(seed=42).permutation(Dataset.shape[0])
    Dataset=Dataset[shuffle,:]
    labels=labels[shuffle,:]
    # one-hot encoding
    labels = to_categorical(labels)
    # split dataset to train,test set
    split_point=0.8
    m=Dataset.shape[0]
    print(int(split_point*m),'!!!!')
    train_x,test_x=Dataset[:int(split_point*m),:],Dataset[int((split_point)*m):,:]
    train_y,test_y=labels[:int(split_point*m),:],labels[int((split_point)*m):,:]
    print(train_x.shape)
    return train_x,train_y,test_x,test_y

class vgg16():

    def build_model(self):
        vgg16=VGG16(weights='imagenet',include_top=False,input_shape=(100,100,3))
        #print(vgg16.summary())
        self.model=Sequential()
        for layer in vgg16.layers:
            layer.trainable=True
            self.model.add(layer)

        self.model.add(Flatten())
        self.model.add(Dense(units=1024))
        self.model.add(Activation(activation='relu'))
        #self.model.add(Dropout(rate=0.6))
        self.model.add(Dense(units=1024))
        self.model.add(Activation(activation='relu'))
        self.model.add(Dense(units=10))
        self. model.add(Activation(activation='softmax'))
        print(self.model.summary())
        self.model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.SGD(lr=0.01, momentum=0.0, decay=0.0, nesterov=False), metrics=['accuracy'])
    def train(self,x_train,y_train):

        history = self.model.fit(x_train, y_train, batch_size=16, epochs=100)
        print(history.history)
        self.model.save_weights(r'C:/Users/Zhiyan/Desktop/CNN_model/vgg16.h5')
        plt.plot(range(len(history.history['loss'])), history.history['loss'])
        plt.show()

    def predict(self,x,y):
        loss,acc=self.model.evaluate(x,y,steps=1,verbose=0)
        print(loss,acc)

train_x,train_y,test_x,test_y=load_data()
print(train_x.shape,test_x.shape)
vgg16=vgg16()
vgg16.build_model()
vgg16.train(train_x,train_y)
#vgg16.model.load_weights(r'C:/Users/Zhiyan/Desktop/CNN_model/vgg16_May20.h5')
vgg16.predict(test_x,test_y)
