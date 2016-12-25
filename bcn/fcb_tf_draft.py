#Created on Fri Oct 28 00:43:02 2016
#author: Luis Campos Garrido
#MessiNeymarIniesta Challenge

#
# TensorFlow Convolutional Network
#

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import glob
from tensorflow.python.framework import ops
from tensorflow.python.framework import dtypes

# Parameters
learning_rate = 0.01
batch_size = 128

# Create the Convolutional Network Model
def conv_net(x):
    # Set weights to be a normal distribution of width*height*depth, 3 color channels
    weights = tf.Variable(tf.random_normal([60*60*3, 3]))
    biases = tf.Variable(tf.random_normal([3]))
    # softmax layer
    out = tf.add(tf.matmul(x, weights), biases)
    return out 

# Load filenames and labels
filenames = []
labels = []
for filename in glob.glob("C:/Users/lcamp/Documents/jobs/methinks/fcbdata/iniesta/*.jpg"):
    filenames.append(filename)
    labels.append(1)
for filename in glob.glob("C:/Users/lcamp/Documents/jobs/methinks/fcbdata/messi/*.jpg"):
    filenames.append(filename)
    labels.append(2)
for filename in glob.glob("C:/Users/lcamp/Documents/jobs/methinks/fcbdata/neymar/*.jpg"):
    filenames.append(filename)
    labels.append(3)
# Shuffle samples
filenames_shuf = []
labels_shuf = []
index_shuf = np.random.permutation(len(filenames))
for i in index_shuf:
    filenames_shuf.append(filenames[i])
    labels_shuf.append(labels[i])
filenames = filenames_shuf
labels = labels_shuf

# Convert string into tensors
filenames_tensor = ops.convert_to_tensor(filenames, dtype=dtypes.string)
labels_tensor = ops.convert_to_tensor(labels, dtype=dtypes.int32)
# Split in training and test sets
split_size = int(len(filenames)*0.75)
X_train, X_test = filenames_tensor[:split_size], filenames[split_size:]
y_train, y_test = labels_tensor[:split_size], labels[split_size:]

# create input queues
input_queue_train = tf.train.slice_input_producer([X_train, y_train], shuffle=False)
input_queue_test = tf.train.slice_input_producer([X_test, y_test], shuffle=False)

# process path and string tensor into an image and a label
image_train = tf.image.decode_jpeg(tf.read_file(input_queue_train[0]), channels=3)
label_train = input_queue_train[1]
image_test = tf.image.decode_jpeg(tf.read_file(input_queue_test[0]), channels=3)
label_test = input_queue_test[1]

# define tensor shape
image_train.set_shape([60, 60, 3])
image_test.set_shape([60, 60, 3])

# collect batches of images before processing
image_batch_train, label_batch_train = tf.train.batch(
                                    [image_train, label_train],
                                    batch_size=50
                                    #,num_threads=1
                                    )
image_batch_test, label_batch_test = tf.train.batch(
                                    [image_test, label_test],
                                    batch_size=50
                                    #,num_threads=1
                                    )

with tf.Session() as sess:
  
  # initialize the variables
  sess.run(tf.initialize_all_variables())
  
  # initialize the queue threads to start to shovel data
  coord = tf.train.Coordinator()
  threads = tf.train.start_queue_runners(coord=coord)

  print("from the train set:")
  for i in range(20):
    print(sess.run(label_batch_train))

  print ("from the test set:")
  for i in range(10):
    print (sess.run(label_batch_test))

  # stop our queue threads and properly close the session
  coord.request_stop()
  coord.join(threads)
  sess.close()

# tf Graph input
x = tf.placeholder(tf.float32, [None, 60*60*3])
y = tf.placeholder(tf.float32, [None, 3])
keep_prob = tf.placeholder(tf.float32) #dropout (keep probability)

# Create the Network
pred = conv_net(x)

# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(pred, y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Evaluate model
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

# Initializing the variables
init = tf.initialize_all_variables()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    image, label = read_images(input_queue)

    total_batch = int(len(X_train)/batch_size)
    for i in range(total_batch):
        batch_x, batch_y = tf.train.batch([image, label], batch_size=batch_size)
        batch_x, batch_y = eval(batch_x), eval(batch_y)
        sess.run([optimizer, cost], feed_dict = {x: batch_x, y: batch_y})
    print("\nTraining complete!")

