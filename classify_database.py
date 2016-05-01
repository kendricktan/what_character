import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import random
from random import randint
from random import shuffle

# Weight initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# Functions for convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')



# Where our handwritten digits are located at
GLOBAL_STEPS = 1000
handwritten_digits_database = './database/letter.data'
trained_session_location = './session/sess_model.ckpt' 

if not os.path.exists('session'):
    os.makedirs('session')
    print('Created session folder')
    print('-----------------------')

if not os.path.exists('graphs'):
    os.makedirs('graphs')
    print('Created graphs folder')
    print('-----------------------')

# Our classifier
print('Reading database...')

# Our dataset
class_list = []
image_list = []

# Opens database
with open(handwritten_digits_database) as database:
    # Loops through each line
    for cur_character in database:
        # Splits delimiter \t
        splitted = cur_character.split('\t') 

        # Structured as: 
        # 1. id
        # 2. letter (a-z)
        # 3. next_id: id for next letter in the word, -1 if last letter
        # 4. word_id: each word is assigned a unique integer id (not used)
        # 5. position of letter in the word (not used)
        # 6. fold 0-9 -- cross-validation fold 
        # 7. p_i_j: 0/1 -- value of pixel in row i, column j

        id = splitted[0]
        letter = ord(splitted[1]) - 97 # Convert to integer
        fold = splitted[5]
        image = splitted[6:-1] # Converts to numpy array

        letter_ = np.zeros((26))
        letter_[letter] = 1

        class_list.append(letter_)
        image_list.append(image)

        # Append our data into our datasets

        # Reshapes our images (to see how it looks like)
        #reconstructed_image = np.reshape(image, (16, 8))

        #print(chr(letter+97))
        #plt.imshow(reconstructed_image, cmap='Greys')
        #plt.show()


# Converts our dataset into numpy array 
class_list = np.array(class_list).astype(np.uint8)
image_list = np.array(image_list).astype(np.float32)

# Constructs our dataset as a numpy array and uses 80% as training data
index_ = int(len(class_list)*0.85)
trnclass_list, partclass_list = class_list[:index_], class_list[index_:]
trnimage_list, partimage_list = image_list[:index_], image_list[index_:]

# Uses 1/3 of testing data as validation data
index__ = int(len(partclass_list)*0.77)
validateclass_list, tstclass_list = partclass_list[:index__], partclass_list[index__:] 
validateimage_list, tstimage_list = partimage_list[:index__], partimage_list[index__:]

print('Building our neural network...')

# Multi layer convolutional neural network
x = tf.placeholder(tf.float32, shape=[None, 128])
y_ = tf.placeholder(tf.float32, shape=[None, 26])
x_image = tf.reshape(x, [-1, 16, 8, 1])

# First layer
W_conv1 = weight_variable([3, 3, 1, 32])
b_conv1 = bias_variable([32])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

# Second layer
W_conv2 = weight_variable([3, 3, 32, 64])
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

# Densely connected layer
W_fc1 = weight_variable([4*2*64, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 4*2*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

# Dropout
keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

# Softmax layer
W_fc2 = weight_variable([1024, 26])
b_fc2 = bias_variable([26])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

# Cost functions
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y_conv), reduction_indices=[1]))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Current session
sess = tf.InteractiveSession()
sess.run(tf.initialize_all_variables())

# Variable to save our session
saver = tf.train.Saver()

# If is is a model we load it
try:
    saver.restore(sess, trained_session_location + '-' + str(GLOBAL_STEPS))
    print('Existing session found, loaded session')

# If there isn't we train our data and save it
except:
    print('Session not found, training data ...')
    print('-----------------------')
    # Train our data
    for i in range(GLOBAL_STEPS):
        # Shuffle data
        perm = np.arange(index_)
        np.random.shuffle(perm)

        temp_trnclass_list = trnclass_list[perm][:100]
        temp_trnimage_list = trnimage_list[perm][:100]

        if i%100 == 0:
            train_accuracy = accuracy.eval(feed_dict={x: trnimage_list, y_: trnclass_list, keep_prob: 1.0})
            print("epoch %d, training accuracy %g"%(i, train_accuracy))

        train_step.run(feed_dict={x: temp_trnimage_list, y_: temp_trnclass_list, keep_prob: 0.5})

    print("Test class accuracy %g"%accuracy.eval(feed_dict={x: tstimage_list, y_: tstclass_list, keep_prob: 1.0}))
    print("Validate class accuracy %g"%accuracy.eval(feed_dict={x: validateimage_list, y_: validateclass_list, keep_prob: 1.0}))

    # Saves our session 
    print('Saving session ...')
    print('-----------------------')
    saver.save(sess, trained_session_location, global_step=GLOBAL_STEPS)

    # Saves visual graph
    print('Saving graph ...')
    print('-----------------------')
    tf.train.SummaryWriter('./graphs', sess.graph)

# Debug
display_classification = True
if display_classification:
    # Grab random piece of data from database
    cur_index = random.randint(0, index__)
    cur_image = validateimage_list[cur_index]
    cur_character = validateclass_list[cur_index]
    reconstructed_image = np.reshape(cur_image, (16, 8))

    # Our predictions
    feed_dict = {x: [cur_image], keep_prob: 1.0}
    predicted_classification = y_conv.eval(feed_dict)

    plt_text = ('predicted: ' + chr(np.argmax(predicted_classification)+97))
    plt_text += (' | actual: ' + chr(np.argmax(cur_character)+97))

    # Displays image
    plt.imshow(reconstructed_image, cmap='Greys')
    plt.title(plt_text)
    plt.show()

sess.close()
