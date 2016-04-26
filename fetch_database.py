import urllib2 
import gzip
import StringIO

letter_data_path = './database/letter.data'
letter_name_path = './database/letter.names'

# Checks if folder exists
if not os.path.exists('database'):
    os.makedirs('database')

# Downloads our dataset
print('Downloading letter.data.gz ...')
letter_data_response = urllib2.urlopen('http://ai.stanford.edu/~btaskar/ocr/letter.data.gz')
compressed_letter_data_gz = StringIO.StringIO(letter_data_response.read())
decompressed_letter_data = gzip.GzipFile(fileobj=compressed_letter_data_gz)
print('Decompressing and writing letter.data to file ...')

# Saves our dataset
with open(letter_data_path, 'w') as outfile:
    outfile.write(decompressed_letter_data.read())
print('Successfully written letter.data in the database folder')

# Downloads additional information about database
print('Downloading letter.names')
letter_name_response = urllib2.urlopen('http://ai.stanford.edu/~btaskar/ocr/letter.names')
compressed_letter_name = StringIO.StringIO(letter_name_response.read())

# Saves our nameset
with open(letter_name_path, 'w') as outfile:
    outfile.write(compressed_letter_name.read())
print('Successfully written letter.names')
