from torch.utils.data import Dataset
from PIL import Image
from sklearn.model_selection import train_test_split
from torchvision import transforms
import pandas as pd
import os
import os.path


class SkinCon(Dataset):
    """ SkinCon Dataset

    Sampled from two datasets:
    fitzpatrick17k: composed of 3230 images
    DDI: composed of 656 images TODO

    """

    def __init__(self, args, train=True, transform=None):
        super(SkinCon, self).__init__()
        self.root = args.dataset_dir  # root folder
        self.train = train  # training set or test set
        self.annotation_file = self.root + "/Fitzpatrick17k/fitzpatrick17k.csv"
        self.annotation_df = pd.read_csv(self.annotation_file)

        # transforms
        self.transforms = transforms.Compose([transforms.Resize((224, 224)),  # 260
                                              transforms.ToTensor()])
        self.transform = transform
        self.count_label_0 = 0

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

        return img, label  # {"image": img, "label": label}

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
        data_dir = self.root + "/Fitzpatrick17k/images"

        for image_name in os.listdir(data_dir):

            # get label from annotations
            label = self.getLabel(image_name)
            labels.append(label)

            # Downsample class 0
            if self.transform == "downsample":
                if label == 0 and self.count_label_0 > 450:
                    continue
                if label == 0:
                    self.count_label_0 += 1

            # read images and apply transform
            image_name = data_dir + "/" + image_name
            image = Image.open(image_name)

            # Upsample class 1 and 2
            if self.transform == "upsample":
                if label == 1 or label == 2:
                    image_aug = image.transpose(Image.FLIP_LEFT_RIGHT)
                    image_aug2 = image.transpose(Image.FLIP_TOP_BOTTOM)
                    image_aug = self.transforms(image_aug)
                    image_aug2 = self.transforms(image_aug2)
                    data.append(image_aug)
                    data.append(image_aug2)
                    labels.append(label)
                    labels.append(label)

            image = self.transforms(image)
            data.append(image)

        return data, labels

    def getLabel(self, image_name):
        """ get the label of the image from the annotation file """

        md5hash = image_name[0:-4]
        label = self.annotation_df[self.annotation_df['md5hash'] == md5hash]['three_partition_label'].item()
        if label == 'non-neoplastic':
            return 0
        elif label == 'benign':
            return 1
        elif label == 'malignant':
            return 2

    def data_train_test_split(self):
        """ split the data into training set and test set """

        all_data = []
        for i in range(len(self.data)):
            all_data.append([self.data[i], self.labels[i]])
        train, val = train_test_split(all_data, random_state=1, train_size=0.7)
        return train, val
