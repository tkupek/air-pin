import glob
import numpy as np
import cv2


def decode_img(img_file, resize):
    img = cv2.imread(img_file)
    if resize is not None:
        img = cv2.resize(img, resize)
    img = np.array(img) / 255.0
    img = (img[:, :, :3])
    img = np.float32(img)
    return img


def get_dataset_files(img_folders, validaton_size=0.2):
    dataset_images = []
    dataset_labels = []
    for i, f in enumerate(img_folders):
        image_files = sorted(glob.glob(f + '/*'))
        dataset_images += image_files
        dataset_labels += [i] * len(image_files)

    ds_size = len(dataset_images)

    # split into training and test set
    np.random.seed(2304)
    validaton_size = int(ds_size * validaton_size)
    selection = np.random.choice(ds_size, validaton_size, replace=False)
    train_images = []
    test_images = []
    train_labels = []
    test_labels = []
    for i in range(ds_size):
        if i not in selection:
            train_images.append(dataset_images[i])
            train_labels.append(dataset_labels[i])
        else:
            test_images.append(dataset_images[i])
            test_labels.append(dataset_labels[i])
    train_images = np.asarray(train_images)
    train_labels = np.asarray(train_labels)
    test_images = np.asarray(test_images)
    test_labels = np.asarray(test_labels)

    return train_images, train_labels, test_images, test_labels


def get_dataset(img_folders, validaton_size=0.2, resize=None):

    train_images, train_labels, test_images, test_labels = get_dataset_files(img_folders, validaton_size)
    train_data = []
    test_data = []

    for img_file in train_images:
        img = decode_img(img_file, resize)
        train_data.append(img)
        train_data.append(np.fliplr(img))
    train_labels = np.repeat(train_labels, 2)
    train_data = np.asarray(train_data)

    for img_file in test_images:
        img = decode_img(img_file, resize)
        test_data.append(img)
        test_data.append(np.fliplr(img))
    test_labels = np.repeat(test_labels, 2)
    test_data = np.asarray(test_data)

    return train_data, train_labels, test_data, test_labels
