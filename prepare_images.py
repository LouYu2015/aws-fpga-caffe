import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import tables

SOURCE_PATH = "data"
TARGET_PATH = "prepared_data"


def normalize_and_rgb(img: np.ndarray):
    '''
    Normalize the image and covert to 3 channels by duplication
    '''
    # normalize image to 0-255 per image.
    img /= np.sum(img)

    # make it rgb by duplicating 3 channels.
    img = np.stack([img, img, img], axis=-1)

    return img


def read_images(fileList):
    '''
    Read images from a list of HDF5 files.

    Yield (image, label)
    '''
    for fileName in fileList:
        with tables.open_file(fileName, 'r') as file:
            for id, img in enumerate(file.root.img_pt):
                yield img, np.argmax(file.root.label[id])


def count_events(fileList):
    '''
    Count the total number of events in the list of files
    '''
    nEvents = 0
    for fileName in fileList:
        with tables.open_file(fileName, 'r') as file:
            nEvents += len(file.root.label)
    return nEvents


def process_images(imgList):
    '''
    Apply pre-processing function on the image and yield (image, label) again
    '''
    for img, label in imgList:
        yield normalize_and_rgb(img), label


def test_read_images():
    files = glob.glob(os.path.join(SOURCE_PATH, "test_*"))
    print("Reading files:")
    print(files)
    print()

    for img, label in process_images(read_images(files)):
        plt.imshow((img/256)**(1/8))
        plt.title(str(label))
        plt.show()


def main():
    import tqdm

    files = glob.glob(os.path.join(SOURCE_PATH, "train_*"))
    for file in files:
        print("Using file:", file)
        nEvents = count_events([file])

        data = read_images([file])
        data = tqdm.tqdm(data, total=nEvents)
        data = process_images(data)

        x = []
        y = []
        for a, b in data:
            x.append(a)
            y.append(b)

        h5file = tables.open_file(os.path.join(TARGET_PATH, os.path.split(file)[-1]), mode="w", title="Test file")
        h5file.create_array("/", "x", x)
        h5file.create_array("/", "y", x)
        input()

if __name__ == '__main__':
    main()
