import cv2
import numpy as np
import matplotlib.pyplot as plt

from pybrain.datasets            import SequenceClassificationDataSet
from pybrain.utilities           import percentError
from pybrain.structure.modules   import LSTMLayer, SoftmaxLayer
from pybrain.supervised          import RPropMinusTrainer
from pybrain.tools.shortcuts     import buildNetwork
from pybrain.datasets            import ClassificationDataSet, SequenceClassificationDataSet

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
        #reconstructed_image = np.reshape(reconstructed_image, (16, 8))

        #plt.imshow(reconstructed_image, cmap='Greys')
        #plt.show()

        # Feed data into our classifcation dataset
        ds.addSample(image, letter)

print('Building our network...')
# Split our data into 75% training and 25% test
tstdata, trndata = ds.splitWithProportion(0.25)

# Converts 1 output to x binary outputs
trndata._convertToOneOfMany()
tstdata._convertToOneOfMany()

# Construct LSTM network
rnn = buildNetwork(trndata.indim, 128, trndata.outdim, hiddenclass=LSTMLayer, outclass=SoftmaxLayer, recurrent=True)
# Our training method
trainer = RPropMinusTrainer(rnn, dataset=trndata, verbose=True)

print('Training our dataset...')
# Carry out the training
trainer.trainEpochs(50)
print ('Percent Error on Test dataset: ' , percentError( trainer.testOnClassData(dataset=tstdata), tstdata['class'] ))

# Save model
NetworkWriter.writeToFile(rnn, 'handwritten_digits_model.xml')

# Plot the first time series
plot(trndata['input'][:, :], '-o')
hold(True)
plot(trndata['target'][:, 0])
show()
