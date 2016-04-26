import cv2
import numpy as np
import matplotlib.pyplot as plt

# Where our handwritten digits are located at
handwritten_digits_database = './database/letter.data'

# Opens file
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
        letter = splitted[1]
        fold = splitted[5]
        image = splitted[6:-1]

        # Reshapes our images (to see how it looks like)
        reconstructed_image = np.array(image).astype(np.uint8)
        reconstructed_image = np.reshape(reconstructed_image, (16, 8))

        plt.imshow(reconstructed_image, cmap='Greys')
        plt.show()

        break

