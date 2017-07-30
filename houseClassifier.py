import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

#loading data
dataframe = pd.read_csv('/vagrant/learning/datasets/housingData/housingData.csv')
dataframe = dataframe.drop(['index','price','sq_price'], axis = 1)
dataframe = dataframe[0:10]

#building labels for training set
dataframe.loc[:,('y1')] = [1,1,1,0,0,1,0,1,1,1]
dataframe.loc[:,('y2')] = dataframe['y1'] == 0
dataframe.loc[:,('y2')] = dataframe['y2'].astype(int)
inputX = dataframe.loc[:, ['area' , 'bathrooms']].as_matrix()
inputY = dataframe.loc[:, ['y1','y2']].as_matrix()

#write out hyperparameters
learning_rate = 0.0000009
training_epochs = 5000
display_step = 50
n_samples = inputY.size

#create neural network
# 2 because of two features
x = tf.placeholder(tf.float32, [None,2])

w = tf.Variable(tf.zeros([2,2])) #create weights
b = tf.Variable(tf.zeros([2])) # adding biases

#multiple weights with inputs
y_values = tf.add(tf.matmul(x,w), b)

#apply softmax to value created
y = tf.nn.softmax(y_values)
#feed matrix of labels
y_ = tf.placeholder(tf.float32, [None,2])

# Perform training
#reduce sum computes the sum of elements across dimentions of the tensor
cost = tf.reduce_sum(tf.pow(y_ -y, 2))/(2*n_samples)
#gradient  descent
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost)

#initialize variables and tensorflow sessions
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

#training loop
for i in range(training_epochs):
	sess.run(optimizer, feed_dict = {x: inputX, y_: inputY})

	#training logs
	if (i) % display_step == 0:
		cc = sess.run(cost, feed_dict ={x: inputX, y_ : inputY})
		print 'Training step:', '%04d' % (i), "cost:", "{:.9f}".format(cc)

print "Optimization Complete"

training_cost = sess.run(cost, feed_dict = {x: inputX, y_: inputY})
print "Training cost:", training_cost, " W:", sess.run(w), " b: ",sess.run(b)


print sess.run(y, feed_dict = {x:inputX})
