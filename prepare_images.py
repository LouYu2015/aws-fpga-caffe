'''
Idea from http://shengshuyang.github.io/hook-up-lmdb-with-caffe-in-python.html
'''
import caffe
import lmdb
import numpy as np
import matplotlib.pyplot as plt
from caffe.proto import caffe_pb2
from caffe.io import datum_to_array, array_to_datum
import os
import glob
import tables

HDF_PATH = "data"


def normalize_and_rgb(images):
    '''
    From https://github.com/nhanvtran/MachineLearningNotebooks/blob/nvt/bwcustomweights-validate/project-brainwave/utils.py
    '''
    import numpy as np
    # normalize image to 0-255 per image.
    image_sum = 1 / np.sum(np.sum(images, axis=1), axis=-1)
    given_axis = 0
    # Create an array which would be used to reshape 1D array, b to have
    # singleton dimensions except for the given axis where we would put -1
    # signifying to use the entire length of elements along that axis
    dim_array = np.ones((1, images.ndim), int).ravel()
    dim_array[given_axis] = -1
    # Reshape b with dim_array and perform elementwise multiplication with
    # broadcasting along the singleton dimensions for the final output
    image_sum_reshaped = image_sum.reshape(dim_array)
    images = images * image_sum_reshaped * 255

    # make it rgb by duplicating 3 channels.
    images = np.stack([images, images, images], axis=-1)

    return images


def read_images(fileList):
    for fileName in fileList:
        with tables.open_file(fileName, 'r') as file:
            for id, img in enumerate(file.root.img_pt):
                yield img, np.argmax(file.root.label[id])


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


def write_images_to_lmdb():
    files = glob.glob(os.path.join(HDF_PATH, "test_*"))
    print(files)
    # for root, dirs, files in os.walk(img_dir, topdown = False):
    #     if root != img_dir:
    #         continue
    #     map_size = 64*64*3*2*len(files)
    #     env = lmdb.Environment(db_name, map_size=map_size)
    #     txn = env.begin(write=True,buffers=True)
    #     for idx, name in enumerate(files):
    #         X = mp.imread(os.path.join(root, name))
    #         y = 1
    #         datum = array_to_datum(X,y)
    #         str_id = '{:08}'.format(idx)
    #         txn.put(str_id.encode('ascii'), datum.SerializeToString())
    # txn.commit()
    # env.close()
    # print " ".join(["Writing to", db_name, "done!"])


if __name__ == '__main__':
    test_read_images()