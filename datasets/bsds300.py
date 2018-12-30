import numpy as np
import h5py
import matplotlib.pyplot as plt

import datasets
import datasets.util


class BSDS300:
    """
    A dataset of patches from BSDS300.
    """

    class Data:
        """
        Constructs the dataset.
        """
        def __init__(self, data):

            self.x = data[:]
            self.N = self.x.shape[0]

    def __init__(self):

        # load dataset
        f = h5py.File(datasets.root + 'BSDS300/BSDS300.hdf5', 'r')

        self.trn = self.Data(f['train'])
        self.val = self.Data(f['validation'])
        self.tst = self.Data(f['test'])

        self.n_dims = self.trn.x.shape[1]
        self.image_size = [int(np.sqrt(self.n_dims + 1))] * 2

        f.close()

    def show_pixel_histograms(self, split, pixel=None):
        """
        Shows the histogram of pixel values, or of a specific pixel if given.
        """

        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        if pixel is None:
            data = data_split.x.flatten()

        else:
            row, col = pixel
            idx = row * self.image_size[0] + col
            data = data_split.x[:, idx]

        n_bins = int(np.sqrt(data_split.N))
        fig, ax = plt.subplots(1, 1)
        ax.hist(data, n_bins, normed=True)
        plt.show()

    def show_images(self, split):
        """
        Displays the images in a given split.
        :param split: string
        """

        # get split
        data_split = getattr(self, split, None)
        if data_split is None:
            raise ValueError('Invalid data split')

        # add a pixel at the bottom right
        last_pixel = -np.sum(data_split.x, axis=1)
        images = np.hstack([data_split.x, last_pixel[:, np.newaxis]])

        # display images
        util.disp_imdata(images, self.image_size, [6, 10])

        plt.show()
