import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

import random
from random import randint
from random import shuffle

# Where our handwritten digits are located at
GLOBAL_STEPS = 1000
handwritten_digits_database = './database/letter.data'
trained_session_location = './session/sess_model.ckpt-' + str(GLOBAL_STEPS)

if not os.path.exists('session'):
    os.makedirs('session')
    print('Created session folder')
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
# Simple softmax regression
x = tf.placeholder(tf.float32, [None, 128])
W = tf.Variable(tf.zeros([128, 26]))
b = tf.Variable(tf.zeros([26]))

y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 26])
cross_entropy = -tf.reduce_sum(y_*tf.log(y))
train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)

init = tf.initialize_all_variables()

correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# Current session
sess = tf.Session()
sess.run(init)

# Variable to save our session
saver = tf.train.Saver()

# If is is a model we load it
try:
    saver.restore(sess, trained_session_location)
    print('Existing session found, loaded session')

# If there isn't we train our data and save it
except:
    # Train our data
    for i in range(GLOBAL_STEPS):
        # Shuffle data
        perm = np.arange(index_)
        np.random.shuffle(perm)

        temp_trnclass_list = trnclass_list[perm][:100]
        temp_trnimage_list = trnimage_list[perm][:100]

        sess.run(train_step, feed_dict={x: temp_trnimage_list, y_: temp_trnclass_list})
        tst_result = ('[Test database] Accuracy: %5.2f%%' % (100*sess.run(accuracy, feed_dict={x: tstimage_list, y_: tstclass_list})))
        validate_result = ('[Validation database] Accuracy: %5.2f%%' % (100*sess.run(accuracy, feed_dict={x: validateimage_list, y_: validateclass_list})))

        print(tst_result + '\t' + validate_result)

    # Saves our session 
    saver.save(sess, trained_session_location, global_step=GLOBAL_STEPS)

# Debug
display_classification = True
if display_classification:
    # Grab random piece of data from database
    cur_index = random.randint(0, index__)
    cur_image = validateimage_list[cur_index]
    cur_character = validateclass_list[cur_index]
    reconstructed_image = np.reshape(cur_image, (16, 8))

    # Our predictions
    feed_dict = {x: [cur_image]}
    predicted_classification = sess.run(y, feed_dict)

    plt_text = ('predicted: ' + chr(np.argmax(predicted_classification)+97))
    plt_text += (' | actual: ' + chr(np.argmax(cur_character)+97))

    # Displays image
    plt.imshow(reconstructed_image, cmap='Greys')
    plt.title(plt_text)
    plt.show()

sess.close()
