import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.io import array_to_datum
import os
import glob
import tables

HDF_PATH = "data"


def normalize_and_rgb(images):
    '''
    From https://github.com/nhanvtran/MachineLearningNotebooks/blob/nvt/bwcustomweights-validate/project-brainwave/utils.py

    Normalize the image and covert to 3 channels by duplication
    '''
    # normalize image to 0-255 per image.
    img /= np.sum(img)

    # make it rgb by duplicating 3 channels.
    img = np.stack([img, img, img], axis=-1)

    return img


def read_images(fileList):
    for fileName in fileList:
        with tables.open_file(fileName, 'r') as file:
            for id, img in enumerate(file.root.img_pt):
                yield img, np.argmax(file.root.label[id])


def count_events(fileList):
    nEvents = 0
    for fileName in fileList:
        with tables.open_file(fileName, 'r') as file:
            nEvents += len(file.root.label)
    return nEvents


def process_images(imgList):
    for img, label in imgList:
        yield normalize_and_rgb(img), label


def test_read_images():
    files = glob.glob(os.path.join(HDF_PATH, "test_*"))
    print("Reading files:")
    print(files)
    print()

    for img, label in process_images(read_images(files)):
        plt.imshow((img/256)**(1/8))
        plt.title(str(label))
        plt.show()


def write_images_to_lmdb(data, db_name):
    env = lmdb.Environment(db_name, map_size=10*1024**3)
    txn = env.begin(write=True, buffers=True)
    for idx, (img, label) in enumerate(data):
        img = np.swapaxes(img, 0, 2)
        datum = array_to_datum(img, label)
        str_id = '{:08}'.format(idx)
        txn.put(str_id.encode('ascii'), datum.SerializeToString())
        if (idx % 1000) == 0:
            txn.commit()
            txn = env.begin(write=True, buffers=True)
        if idx > 100:
            break
    txn.commit()
    env.close()


def main():
    import tqdm

    files = glob.glob(os.path.join(HDF_PATH, "train_*"))
    print("Using files:", files)
    nEvents = count_events(files)
    print("Number of events:", nEvents)

    data = read_images(files)
    print("Writing to database")
    data = tqdm.tqdm(data, total=nEvents)
    data = process_images(data)
    write_images_to_lmdb(data, "train.mdb")


if __name__ == '__main__':
    main()