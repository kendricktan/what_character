import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pybrain.utilities import percentError
from pylab import plot, hold, show
from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer, BackpropTrainer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import ClassificationDataSet, SequenceClassificationDataSet
from pybrain.tools.xml.networkwriter import NetworkWriter
from pybrain.tools.xml.networkreader import NetworkReader

# Where our handwritten digits are located at
handwritten_digits_database = './database/letter.data'

# Our classifier
ds = ClassificationDataSet(128, 1, nb_classes=26)

print('Reading database...')
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
        letter = ord(splitted[1]) - 97 # Convert to interger
        fold = splitted[5]
        image = np.array(splitted[6:-1]).astype(np.uint8) # Converts to numpy array

        # Reshapes our images (to see how it looks like)
        #reconstructed_image = np.reshape(image, (16, 8))

        #print(chr(letter+97))
        #plt.imshow(reconstructed_image, cmap='Greys')
        #plt.show()

        # Feed data into our classifcation dataset
        ds.addSample(image, letter)

print('Building our recurrent neural network...')
# Split our data into 75% training and 25% test
trndata, partdata = ds.splitWithProportion(0.75)
tstdata, validata = partdata.splitWithProportion(0.5)

# Converts 1 output to x binary outputs
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()
validata._convertToOneOfMany()

# Construct LSTM network
rnn = buildNetwork(trndata.indim, 128, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, recurrent=True)
# Our training method
trainer = BackpropTrainer( rnn, dataset=trndata, verbose=True, momentum=0.9, learningrate=0.00001  )

if os.path.isfile('ocr_model.xml'):
    print('Existing network model found, loading model...')
    rnn = NetworkReader.readFrom('ocr_model.xml')
    print('Loaded model')

else:
    print('Training our dataset...')
    # Carry out the training
    trainer.trainOnDataset(trndata, 50)
    print('Total epochs: ' + str(trainer.totalepochs))

    # Save model
    NetworkWriter.writeToFile(rnn, 'ocr_model.xml')
    print('Saved neural networl model')

# Plot the first time series
predict = rnn.activateOnDataset(tstdata).argmax(axis=1)
print('Error: ' + str(percentError(predict, tstdata['class'])*100))
