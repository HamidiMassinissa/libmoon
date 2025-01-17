import torch
import pickle
from sklearn.model_selection import train_test_split
# taken from https://github.com/Xi-L/ParetoMTL and adapted
import os
import numpy as np

from libmoon.util_global.constant import root_name


class DspritesData(torch.utils.data.Dataset):
    """
    The datasets from ParetoMTL
    """
    def __init__(self, dataset, split, labels_ids: list[int]=[1,2], root='data/multi', **kwargs):
        assert dataset in ['dsprites']
        assert split in ['train', 'val', 'test']

        # equal size of val and test split
        train_split = .9

        if dataset == 'dsprites':
            self.path = os.path.join(root_name, 'problem', 'mtl', 'data', 'dsprites', 'dsprites_ndarray_co1sh3sc6or40x32y32_64x64.npz')

        self.test_size = 0.1
        self.val_size = 0.1
        with open(self.path, 'rb') as f:
            with np.load(f) as data:
                trainX = data['imgs']
                trainLabel = data['latents_classes'][:,labels_ids]

        n_train = len(trainX)
        if self.test_size > 0:
            trainX, testX, trainLabel, testLabel = train_test_split(
                trainX, trainLabel, test_size=self.test_size, random_state=42
            )
            n_train = len(trainX)
            n_test = len(testX)

        if self.val_size > 0:
            trainX, valX, trainLabel, valLabel = train_test_split(
                trainX, trainLabel, test_size=self.val_size, random_state=42
            )
            n_train = len(trainX)
            n_val = len(valX)

        if split in ['train', 'val']:
            if split == 'val':
                if self.val_size > 0:
                    valX = torch.from_numpy(valX.reshape(n_val, 1, 64, 64)).float()
                    valLabel = torch.from_numpy(valLabel).long()
                self.X = valX
                self.y = valLabel
            elif split == 'train':
                trainX = torch.from_numpy(trainX.reshape(n_train, 1, 64, 64)).float()
                trainLabel = torch.from_numpy(trainLabel).long()
                self.X = trainX
                self.y = trainLabel
        elif split == 'test':
            testX = torch.from_numpy(testX.reshape(n_test, 1, 64, 64)).float()
            testLabel = torch.from_numpy(testLabel).long()
            self.X = testX
            self.y = testLabel

    def __getitem__(self, index):
        return dict(data=self.X[index], labels_1=self.y[index, 0], labels_2=self.y[index, 1])

    def __len__(self):
        return len(self.X)

    def task_names(self):
        return ['1', '2']







if __name__ == '__main__':
    import matplotlib.pyplot as plt
    dst = DspritesData(dataset='dsprites', split='val')
    loader = torch.utils.data.DataLoader(dst, batch_size=10, shuffle=True, num_workers=0)

    for dat in loader:
        ims = dat['data'].view(10, 36, 36).numpy()
        labs_l = dat['labels_1']
        labs_r = dat['labels_2']

        f, axarr = plt.subplots(2, 5)
        for j in range(5):
            for i in range(2):
                axarr[i][j].imshow(ims[j * 2 + i, :, :], cmap='gray')
                axarr[i][j].set_title('{}_{}'.format(labs_l[j * 2 + i], labs_r[j * 2 + i]))
        plt.show()
        a = input()

        if a == 'ex':
            break
        else:
            plt.close()

