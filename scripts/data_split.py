#!/usr/bin/env python3


from pathlib import Path
import os
import random


DEFAULT_OUTPUT_PATH = '/zhome/25/e/155273/Desktop/02516_dvcv/02516_intro_cnn_w2'


def write_to_file(folders, file_name):
    with open(file_name, 'w') as f:
        for folder in folders:
            f.write("%s\n" % folder)


def split_data(path, output_path=DEFAULT_OUTPUT_PATH):
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

    # Write to respective files
    write_to_file(train_folders, output_path / 'train.txt')
    write_to_file(test_folders, output_path / 'test.txt')
    write_to_file(validation_folders, output_path / 'validation.txt')


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--path',
        type=Path,
        default='/dtu/datasets1/02516/PH2_Dataset_images',
    )
    parser.add_argument(
        '--output-path',
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
    )
    args = parser.parse_args()

    split_data(args.path, output_path=args.output_path)


if __name__ == '__main__':
    main()
