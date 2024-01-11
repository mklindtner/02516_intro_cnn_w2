# path = '/dtu/datasets1/02516/PH2_Dataset_images'

import os
import random

# Path to the dataset
path = "/dtu/datasets1/02516/PH2_Dataset_images"

# List all folders in the path
folders = [f for f in os.listdir(path) if os.path.isdir(os.path.join(path, f))]

# Shuffle the folder list
random.shuffle(folders)

# Split ratios for train, test, validation
	train_ratio = 0.7
	test_ratio = 0.15
# Validation ratio is implied

# Calculate split indices
	total_folders = len(folders)
	train_end = int(train_ratio * total_folders)
	test_end = train_end + int(test_ratio * total_folders)

# Split the list
	train_folders = folders[:train_end]
	test_folders = folders[train_end:test_end]
	validation_folders = folders[test_end:]

# Function to write folders to a file
	def write_to_file(folders, file_name):
		    with open(file_name, 'w') as f:
					        for folder in folders:
									            f.write("%s\n" % folder)

# Write to respective files
write_to_file(train_folders, '/zhome/25/e/155273/Desktop/02516_dvcv/02516_intro_cnn_w2/train.txt')
write_to_file(test_folders, '/zhome/25/e/155273/Desktop/02516_dvcv/02516_intro_cnn_w2/test.txt')
write_to_file(validation_folders, '/zhome/25/e/155273/Desktop/02516_dvcv/02516_intro_cnn_w2/validation.txt')
