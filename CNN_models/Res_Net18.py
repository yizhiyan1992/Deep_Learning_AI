import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Flatten,Input,Conv2D,ZeroPadding2D,MaxPooling2D,AveragePooling2D,BatchNormalization,Activation,GlobalAveragePooling2D,Dense
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.utils import plot_model
from scipy import misc
import os
print(tf.__version__)
os.chdir(r'C:/Users/Zhiyan/Desktop/Sign-Language-Digits-Dataset/')

class ResNet18():
    def __init__(self):
        #self.model=Model()
        inputs=Input(shape=(224,224,3))
        x=ZeroPadding2D((3,3))(inputs)
        x=Conv2D(filters=64,kernel_size=(7,7),strides=(2,2))(x)
        x=BatchNormalization(axis=-1)(x)
        x=Activation(activation='relu')(x)
        #x = ZeroPadding2D((1,1))(x)
        x=MaxPooling2D(pool_size=(3,3),strides=(2,2))(x)
        x=Basic_block(64,[1,1],1)(x)
        x = Basic_block(64,[1,1],1)(x)
        x = Basic_block(128,[2,1],2)(x)
        x = Basic_block(128,[1,1],1)(x)
        x = Basic_block(256,[2,1],2)(x)
        x = Basic_block(256,[1,1],1)(x)
        x = Basic_block(512,[2,1],2)(x)
        x = Basic_block(512,[1,1],1)(x)
        x=GlobalAveragePooling2D()(x)
        x=Flatten()(x)
        x=Dense(units=10,activation='softmax')(x)
        self.model=tf.keras.Model(inputs=inputs,outputs=x)
        #plot_model(self.model,r'C:/Users/Zhiyan/Desktop/CNN_model/ResNet18.png',show_shapes=True)
        print(self.model.summary())

    def train(self,x,y):
        self.model.compile(loss='categorical_crossentropy',optimizer='rmsprop',metrics=['accuracy'])
        history=self.model.fit(x=x,y=y,batch_size=16,epochs=100)
        self.loss=history.history['loss']

    def test(self,x,y):
        loss,ac=self.model.evaluate(x,y)
        print(loss,ac)

    def plot_loss(self):
        print(self.loss)
        plt.plot(range(len(self.loss)),self.loss)
        plt.show()

class Basic_block(tf.keras.layers.Layer):
    def __init__(self,filters,stride=[1,1],res_stride=1):
        super().__init__()
        self.conv1=Conv2D(filters=filters,kernel_size=(3,3),padding='same',strides=stride[0])
        self.BN1=BatchNormalization(axis=-1)
        self.Relu1=Activation(activation='relu')
        self.conv2 = Conv2D(filters=filters, kernel_size=(3, 3), padding='same',strides=stride[1])
        self.BN2 = BatchNormalization(axis=-1)
        self.Relu2 = Activation(activation='relu')
        self.add=tf.keras.layers.Add()
        self.downsampleConv=Conv2D(filters=filters,strides=(2,2),padding='same',kernel_size=(1,1))
        self.downsampleBN=BatchNormalization(axis=-1)
        self.stride=res_stride
        self.Relu3=Activation(activation='relu')

    def call(self,inputs,training=None):
        if self.stride==1:
            res=inputs
        else:
            res=self.downsampleConv(inputs)
            res=self.downsampleBN(res,training=training)
        x=self.conv1(inputs)
        x=self.BN1(x,training=training)
        x=self.Relu1(x)
        x=self.conv2(x)
        x=self.BN2(x,training=training)
        x=self.add([x,res])
        output=self.Relu3(x)
        return output


files=os.listdir('Dataset/')
print(files)
def load_data():
    for file_idx,file in enumerate(files):
        cur_add='Dataset/'+file+'/'
        pics=os.listdir(cur_add)
        for idx,pic in enumerate(pics):
            # set the number of samples
            #if idx==10: break
            if file_idx==0 and idx==0:
                picture=misc.imread(cur_add+pic)
                picture= misc.imresize(picture, size=(224, 224, 3))
                picture=picture/255
                picture = picture[np.newaxis, :]
                Dataset=picture
                labels=[[idx]]
                continue
            picture=misc.imread(cur_add+pic)
            picture = misc.imresize(picture, size=(224, 224, 3))
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


def main():
    train_x,train_y,test_x,test_y=load_data()
    #plt.imshow(train_x[0,:])
    #plt.show()
    cnn=ResNet18()
    cnn.train(train_x,train_y)
    cnn.test(test_x,test_y)
    cnn.plot_loss()

if __name__=='__main__':
    main()
