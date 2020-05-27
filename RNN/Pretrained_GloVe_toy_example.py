from tensorflow.keras.datasets import imdb
from tensorflow.keras import preprocessing
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten,Dense,Embedding

#download raw data from aclIMDB
import os
imdb_dir=r'C:/Users/Zhiyan/Desktop/aclImdb/aclImdb/'
train_dir=os.path.join(imdb_dir,'train')
labels=[]
texts=[]

for label_type in ['neg','pos']:
    dir_name=os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:]=='.txt':
            f=open(os.path.join(dir_name,fname),encoding='utf-8')
            texts.append(f.read())
            f.close()
            if label_type=='neg':
                labels.append(0)
            else:
                labels.append(1)

#tokenization
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
maxlen=100
training_samples=200
validation_samples=10000
max_words=10000
tokenizer=Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequence=tokenizer.texts_to_sequences(texts)
word_index=tokenizer.word_index
print('found %s unique tokens.' %len(word_index))

#change the raw data (tokens) into index, and zero-pad each sample
#shuffle data train/val split
data=pad_sequences(sequence,maxlen=maxlen)
labels=np.asarray(labels)
print(data.shape,labels.shape)
indices=np.arange(data.shape[0])
np.random.shuffle(indices)
data=data[indices]
labels=labels[indices]
x_train=data[:training_samples]
y_train=labels[:training_samples]
x_val=data[training_samples:training_samples+validation_samples]
y_val=labels[training_samples:training_samples+validation_samples]

# download pretrained GloVe vector
glove_dir='/Users/Zhiyan/Desktop/glove.6B'
embeddings_index={}
f=open(os.path.join(glove_dir,'glove.6B.100d.txt'),encoding='utf-8')
for line in f:
    values=line.split()
    word=values[0]
    coefs=np.asarray(values[1:],dtype='float32')
    embeddings_index[word]=coefs
f.close()
print(len(embeddings_index))

#preparing the GloVe word-embedding matrix
embedding_dim=100
embedding_matrix=np.zeros((max_words,embedding_dim))
for word,i in word_index.items():
    if i<max_words:
        embeddings_vector=embeddings_index.get(word)
        if embeddings_vector is not None:
            embedding_matrix[i]=embeddings_vector

#model definition
model=Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()
#loading pretrained word embeddings into the Embedding layer
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False
#train & evaluating
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history=model.fit(x_train,y_train,epochs=10,batch_size=32,validation_data=(x_val,y_val))

print(history.history)
#plotting the results
import matplotlib.pyplot as plt
acc=history.history['acc']
val_acc=history.history['val_acc']
loss=history.history['loss']
val_loss=history.history['val_loss']
epochs=range(1,len(acc)+1)
plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and Validation accuracy')
plt.legend()

plt.figure()
plt.plot(epochs,loss,'bo',label='Training acc')
plt.plot(epochs,val_loss,'b',label='Validation acc')
plt.title('Training and Validation loss')
plt.legend()
plt.show()
