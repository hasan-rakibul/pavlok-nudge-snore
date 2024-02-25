import os
import random
import shutil
import argparse

def split_and_move(data_dir, val_ratio=0.2, test_ratio=0.1):
    '''Split the dataset into val and test sets, and move them to the corresponding directories.
    The source directory is the train directory, from where we move the files to val and test directories.'''

    input_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'val')
    test_dir = os.path.join(data_dir, 'test')

    for class_name in os.listdir(input_dir):
        class_dir = os.path.join(input_dir, class_name)
        if not os.path.isdir(class_dir):
            continue

        val_class_dir = os.path.join(val_dir, class_name)
        if not os.path.exists(val_class_dir):
            os.makedirs(val_class_dir)
        test_class_dir = os.path.join(test_dir, class_name)
        if not os.path.exists(test_class_dir):
            os.makedirs(test_class_dir)

        file_list = os.listdir(class_dir)
        total_sample = len(file_list)
        print(f'{class_name} class: {total_sample} total samples')

        num_val = int(total_sample * val_ratio)
        num_test = int(total_sample * test_ratio)
        print(f'(Expected) Class {class_name}: {num_val} val samples, {num_test} test samples')

        val_samples = random.sample(file_list, num_val)
        file_list = list(set(file_list) - set(val_samples))
        test_samples = random.sample(file_list, num_test)

        for file in val_samples:
            shutil.move(os.path.join(class_dir, file), os.path.join(val_class_dir, file))
        for file in test_samples:
            shutil.move(os.path.join(class_dir, file), os.path.join(test_class_dir, file))

        print(f'\tClass {class_name} splitted succuessfully:')
        print(f'\t{len(os.listdir(class_dir))} train samples, {len(os.listdir(val_class_dir))} val samples, {len(os.listdir(test_class_dir))} test samples\n\n')

def main():
    parser = argparse.ArgumentParser(description='Split the dataset into train, val and test sets.')
    parser.add_argument('--data_dir', type=str, default='./data/', help='Path to the dataset directory')
    parser.add_argument('--val_ratio', type=float, default=0.2, help='Ratio of validation samples')
    parser.add_argument('--test_ratio', type=float, default=0.1, help='Ratio of test samples')

    args = parser.parse_args()
    split_and_move(data_dir=args.data_dir, val_ratio=args.val_ratio, test_ratio=args.test_ratio)

if __name__ == '__main__':
    main()