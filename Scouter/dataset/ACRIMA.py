from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import os
import os.path


class ACRIMA(Dataset):
    """  ACRIMA Dataset

    It is composed of 396 glaucomatous images and 309 normal images.

    TODO

    """

    # classes = [0, 1]  # binary classification 0: normal 1: glaucomatous

    def __init__(self, args, train=True, transform=None):
        super(ACRIMA, self).__init__()
        self.root = args.dataset_dir    # root folder
        self.train = train  # training set or test set

        # transforms
        self.transforms = transforms.Compose([transforms.Resize((260, 260)),
                                              transforms.ToTensor()])

        self.data, self.labels = self.getData()
        self.train, self.val = self.data_train_test_split()

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        if self.train:
            img, label = self.train[index]
        else:
            img, label = self.val[index]

        return {"image": img, "label": label}

    def __len__(self):
        if self.train:
            return len(self.train)
        else:
            return len(self.val)

    def getData(self):
        """
        Read data from the dataset dir.
        Returns:
            data: images
            labels: labels of the data
        """

        data = []
        labels = []
        data_dir = os.path.join(self.root, "Images")

        for image_name in os.listdir(data_dir):

            # get label from the image_name
            label = self.getLabel(image_name)
            labels.append(label)

            # read images and apply transform
            image_name = data_dir + "/" + image_name
            image = Image.open(image_name)
            image = self.transforms(image)
            # image = transforms.ToPILImage()(image)  # TODO
            data.append(image)

        return data, labels

    @staticmethod
    def getLabel(image_name):
        """ get the label of the image """

        if '_g_' in image_name:
            label = 1
        else:
            label = 0
        return label

    def data_train_test_split(self):
        """ split the data into training set and test set """

        all_data = []
        for i in range(len(self.data)):
            all_data.append([self.data[i], self.labels[i]])
        train, val = train_test_split(all_data, random_state=1, train_size=0.7)
        return train, val

