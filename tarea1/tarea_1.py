"""
Tarea 1 CC5508 Procesamiento y Análisis de imágenes
El archivo toma una imagen y codifica/decodifica un texto en ella
"""

import skimage.io as skio
import numpy as np
import argparse


def to_uint8(image):
	"""

	:param image:
	:return:
	"""
	if image.dtype == np.float64:
		image = image * 255
	image[image < 0] = 0
	image[image > 255] = 255
	image = image.astype(np.uint8, copy=False)
	return image


def imread(filename, as_gray=False):
	"""

	:param filename:
	:param as_gray:
	:return:
	"""
	image = skio.imread(filename, as_gray=as_gray)
	if image.dtype == np.float64:
		image = to_uint8(image)
	return image


def get_ascii_representation(char):
	"""
	Get ASCII number from char
	:param char:
	:return:
	"""
	return ord(char)


def get_binary_representation(num, bits):
	"""
	Convert ASCII number to binary representation.
	:param num:
	:param bits:
	:return:
	"""

	# A pixel has only 8 bits
	assert 1 <= bits <= 8

	# We get binary representation as 0bXXXX -> XXXX
	bin_rep = str(bin(num))[2:]
	
	# ASCII characters have at least 7 bits
	# In order to get equally distributed bits
	# we fill with zeros until bits divide len(bin_rep)
	# (starting from len = 7)
	while len(bin_rep) % bits != 0 or len(bin_rep) < 7:
		bin_rep = '0' + bin_rep
	return bin_rep


def encode(image, textfile, bits):
	"""
	Hide some text into an image using Steganography
	:param image:
	:param textfile:
	:param bits:
	:return:
	"""

	print('Starting encoding for '+str(image))

	# Read Image
	print('Reading '+str(image))	
	img = imread(image)
	
	# Read text from file
	print('Reading '+str(textfile))
	with open(textfile, 'r') as file:
		text = file.read()
	print('Message is: '+str(text))

	# Encode text to binary ASCII
	print('Encoding text')
	
	# We vectorize methods (we will use numpy arrays)
	to_ascii = np.vectorize(get_ascii_representation)
	to_binary = np.vectorize(get_binary_representation)
	
	# Getting an array of characters from text
	array_text = np.array(list(text))
	
	# Getting an array of ASCII integers from char array
	ascii_text = to_ascii(array_text)
	
	# Converting every integer to binary represention
	encoded_text = to_binary(ascii_text, bits)
	
	# We concatenate every binary string
	# It makes encoding easier
	bin_rep_text = ''
	for bin_char in encoded_text:
		bin_rep_text += bin_char
		
	# We put a zero at the end, so we know text ends there
	encoded_text = bin_rep_text + get_binary_representation(0, bits)
	
	# Encoded text looks like this:
	# encoded_text -> array(['a', 'b', 'c'])
	# encoded_text -> array([97,98,99])
	# encoded_text -> array(['1100001', '1100010', '1100011'], dtype='<U7')
	# encoded_text -> '1100001110001011000110000000'
	
	# We initialize pointer
	k = 0      # Actual bit
	
	# Read every pixel and RGB channel, encoding text
	try:
		for i in range(0,len(img)):
			for j in range(0, len(img[0])):
				for l in range(0,3):
					
					# If we run out of bits, we stop
					if k >= len(encoded_text):
						raise Exception('End of encoding')
						
					# Extracting RGB binary values as binary
					rgb_value = get_binary_representation(img[i,j,l], bits)
					
					# Appending <bits> from encoded_text
					rgb_value = rgb_value[0:len(rgb_value)-bits] + encoded_text[k:k+bits]
					
					# Updating channel value
					img[i,j,l] = int(rgb_value, 2)
					
					# Moving pointer
					k += bits
					
	except Exception as error:
		print('Encode was successful')
	
	# Store number of bits in channel R, last pixel
	img[len(img)-1, len(img[0])-1,0] = bits
	
	# Renaming image
	image = image.replace(".jpg", "")
	image = image.replace(".png", "")
	image = image+"_out.png"
	
	# Save new Image
	print('Saving Image')
	skio.imsave(image, img)


def decode(image):
	"""

	:param image:
	:return:
	"""

	# Read Image
	img = skio.imread(image, as_gray = False)
	
	# Get number of bits used on encoding
	bits = img[len(img)-1, len(img[0])-1,0]
	
	# Get number of bits to read
	# If we read len(bin_rep), then we know it is a char
	bits_already_read = 0
	bits_to_read = 7
	
	# Same process as before
	while bits_to_read % bits != 0:
		bits_to_read += 1
	
	# Initialize empty string:
	text = ''
	aux = ''
	
	# Read every pixel and RGB channel, hidding text
	try:
		for i in range(0,len(img)):
			for j in range(0, len(img[0])):
				for l in range(0,3):
				
					# Extracting RGB binary values as binary
					rgb_value = get_binary_representation(img[i,j,l], bits)
					
					# Extracting last <bits>
					rgb_value = rgb_value[len(rgb_value)-bits:len(rgb_value)]
					
					# Reconstructing binary representation
					aux += rgb_value
					bits_already_read += bits
					
					# If we already have a char
					if bits_already_read == bits_to_read:
					
						# Converting binary to integer
						aux = int(aux, 2)
						
						# Converting int to char
						character = chr(aux)
						
						# If it is end of the line, we stop
						if character == '\x00':
							raise Exception('End of decoding')
							
						# We reconstruct the text
						text += character
						bits_already_read = 0
						aux = ''
	
	except Exception as error:
		print('Decoding was successful')
		
	print("Message is: "+str(text))


if __name__ == '__main__': 
	parser = argparse.ArgumentParser(description = "Encoding text on images")
	parser.add_argument("--encode", action='store_true')
	parser.add_argument("--decode", action='store_true')
	parser.add_argument("--image", type=str, help="Image file", required = True)
	parser.add_argument("--text", type=str, help="Text file", required = False)
	parser.add_argument("--nbits", type=int, help="Number of bits", required = False)
	
	pargs = parser.parse_args()
	
	if pargs.encode and pargs.decode:
		print('Please choose only one')
	elif pargs.encode:
		if not pargs.nbits or not pargs.text:
			print('Error: Text file and number of bits required')
		else:
			encode(pargs.image, pargs.text, pargs.nbits)
	elif pargs.decode:
		decode(pargs.image)