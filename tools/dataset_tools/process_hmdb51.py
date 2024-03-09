import os
import sys


def process_HMDB51_data(data_path, list_path, split_index):
    train_videos = []
    test_videos = []
    action_classes = {}

    # First, collect all unique action classes and assign an index to each
    for file_name in os.listdir(list_path):
        if file_name.endswith(
                ".txt") and f'_test_split{split_index}.txt' in file_name:
            action_class = file_name.split('_')[0]
            if action_class not in action_classes:
                action_classes[action_class] = len(action_classes)
        else:
            print(f'Error file: {file_name}')
    print(len(action_classes))
    # Process the videos and assign the class index based on action class
    for file_name in os.listdir(list_path):
        if file_name.endswith(
                ".txt") and f'_test_split{split_index}.txt' in file_name:
            print(f'Processing {file_name}.')
            with open(os.path.join(list_path, file_name), 'r') as file:
                lines = file.readlines()
                action_class = file_name.split('_')[0]
                class_index = action_classes[action_class]  # Get class index
                for line in lines:
                    video_name, label = line.strip().split()
                    video_path = os.path.join(f'{action_class}/{video_name}')
                    video_entry = f'{video_path} {class_index}\n'  # Include class index

                    if label == '1':
                        train_videos.append(video_entry)
                    elif label == '2':
                        test_videos.append(video_entry)

    # Ensure output paths for CSVs
    train_file_path = os.path.join(data_path, f'train_split{split_index}.csv')
    test_file_path = os.path.join(data_path, f'test_split{split_index}.csv')
    val_file_path = os.path.join(data_path, f'val_split{split_index}.csv')

    # Write the CSV files
    with open(train_file_path, 'w') as f:
        f.writelines(train_videos)

    with open(test_file_path, 'w') as f:
        f.writelines(test_videos)

    with open(val_file_path, 'w') as f:
        f.writelines(test_videos)

    # Create symbolic links for convenience
    train_link_path = os.path.join(data_path, 'train.csv')
    test_link_path = os.path.join(data_path, 'test.csv')
    val_link_path = os.path.join(data_path, 'val.csv')

    if not os.path.exists(train_link_path):
        os.symlink(train_file_path, train_link_path)
    if not os.path.exists(test_link_path):
        os.symlink(test_file_path, test_link_path)
    if not os.path.exists(val_link_path):
        os.symlink(val_file_path, val_link_path)

    # Print symlink creation log
    print(f'Created symbolic link: {train_link_path} <- {train_file_path}')
    print(f'Created symbolic link: {test_link_path} <- {test_file_path}')
    print(f'Created symbolic link: {val_link_path} <- {val_file_path}')


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python script.py <data_path> <list_path> <split_index>")
        sys.exit(1)

    data_path = sys.argv[1]
    list_path = sys.argv[2]
    split_index = int(sys.argv[3])

    os.makedirs(list_path, exist_ok=True)
    process_HMDB51_data(data_path, list_path, split_index)
