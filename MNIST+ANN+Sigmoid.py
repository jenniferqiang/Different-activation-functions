# -*- coding: utf-8 -*-
"""
Created on Tue Jan 22 15:31:25 2019

@author: zh846675
"""
import pandas as pd
import numpy as np
df=pd.read_csv(r'C:\Users\zh846675\MNIST\Matrix-200.csv').drop(['Unnamed: 0'],axis=1)

#%%
A=df.as_matrix()
A=A.transpose()
#%%
import tensorflow as tf
from keras.utils import np_utils

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

#%%
input_shape = (784)
# Making sure that the values are float so that we can get decimal points after division
A=A.astype('float32')
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
# Normalizing the RGB codes by dividing it to the max RGB value.
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('Number of images in x_train', x_train.shape[0])
print('Number of images in x_test', x_test.shape[0])
#%%
#parameters
x_train=x_train.reshape(60000,784)
x_test=x_test.reshape(len(x_test),784)
#y_train=y_train.reshape(60000,1)
#y_train = np_utils.to_categorical(y_train, 10)
#y_test = np_utils.to_categorical(y_test, 10)
#%%
#Parameters
learning_rate=0.005
num_steps=60000
batch_size=100
display_step=1000
#Network Parameters
n_hidden_1=400
n_hidden_2=800
num_input=784
num_classes=10
#%%
X=tf.placeholder("float32",[None, 784])
Y=tf.placeholder("float32",[None,])
#XX=tf.reshape(X,[-1,784])
#%%
weights={       
    'h1':tf.Variable(tf.truncated_normal([num_input, n_hidden_1], stddev=0.1, seed=0)),
    'out':tf.Variable(tf.truncated_normal([n_hidden_1,num_classes], stddev=0.1, seed=0))        
        }
biases = {
    'b1': tf.Variable(tf.truncated_normal([n_hidden_1],stddev=0.1, seed=0)),
    'out': tf.Variable(tf.truncated_normal([num_classes],stddev=0.1, seed=0))
        }
#%%
act=tf.matmul(X,weights['h1'])+biases['b1']
#act2=tf.matmul(act,A)
a1=tf.nn.sigmoid(act)
act1=tf.matmul(a1,weights['out'])+biases['out']
prediction=tf.nn.softmax(act1)
#%%
labels=tf.to_int64(Y)
loss_op = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=prediction)
#cross_entropy = -tf.reduce_mean(Y*tf.log(prediction))*1000.0
optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss_op)

#%%
# accuracy of the trained model, between 0 (worst) and 1 (best)
correct_prediction = tf.equal(tf.argmax(prediction,1),labels)
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
#%%
#Initialize the variables
#import numpy as np

def next_batch(batch_size, data, labels,s):
    '''
    Return a total of `num` random samples and labels. 
    '''
    data_step=data[batch_size*s:batch_size*s+batch_size]
    labels_step=labels[batch_size*s:batch_size*s+batch_size]
    return np.asarray(data_step), np.asarray(labels_step)

#%%
#Start training
weight=[]
mid=[]
#clip_op= tf.assign(weights['h1'],tf.clip_by_value(weights['h1'],0,1))
with tf.Session() as sess:
    s=0
    #writer=tf.summary.FileWriter(r'C:\Users\zh846675\Project2\Demo1\Step Function\graphs',sess.graph)
    init=tf.global_variables_initializer() 
    sess.run(init)
    #sess.run(tf.assign(weights['h1'],tf.clip_by_value))
    for step in range(1, num_steps+1):
        batch_x, batch_y=next_batch(batch_size,x_train,y_train,s)
        if s<600:
            s=s+1
        else:
            s=0
        #Run optimization op (backprop)
        sess.run(optimizer, feed_dict={X: batch_x, Y: batch_y})
        #sess.run(clip_op)
        if step % display_step==0 or step==1:
            loss, acc, w, b=sess.run([loss_op, accuracy, weights, biases], feed_dict={X: batch_x, Y: batch_y})
            weight.append(w)
            print("Step "+str(step)+", Minibatch Loss=" + "{:.4f}".format(loss)+", Training Accuracy="+"{:.3f}".format(acc))
    print ("Optimization Finished!")
        #Calculate accuracy for MNIST test images
    print ("Training Accuracy:", sess.run([loss_op,accuracy], feed_dict={X: x_train, Y: y_train}))
    print ("Testing Accuracy:", sess.run([loss_op,accuracy], feed_dict={X: x_test, Y: y_test}))
    print ("Sigmoid")
#writer.close()




