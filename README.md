# Deep_Learning_AI
#ANN folder:
AndrewNg's Coursera class
Task 1:
  build a L-layer Neural Network for binary classification problem (0-1)
  Dataset: bank-note folder (from UC-Irvine)

Task 2:
build a L-layer Neural Network with L2-regularization
Same dataset from task 1 is used in this file, except that only 2 features are used for visualization purpose. L2 norm is controlled by the parameter lambd.
Three different lambds are testes, namely [0, 0.01, 0.5]
The plots (4 figures) indicate their loss values with first 5000 iterations (pic 1); their prediction regions (pic2-pic4)
The result indicates that prediction accuracy on test set is higher if using L2-normalization.

Task 3:
build a L-layer Neural Network with Drop-out regularization
Three different save_prob are testes, namely [1, 0.95, 0.8]
The plots (4 figures) indicate their loss values with first 15000 iterations (pic 1); their prediction regions (pic2-pic4). Result indicates that when dropout_rate=0.8 it performs the best on test set.

#CNN folder:
target: using CNN model to recognize the hand signs of numbers
1. LeNet5 model
2. Pretrained VGG16 model
3. ResNet18 model

#RNN folder:
1. pretrained GloVe word embedding.
download text dataset from:
http://mng.bz/0tIo
download GloVe word embedding from:
http://nlp.stanford.edu/projects/glove
