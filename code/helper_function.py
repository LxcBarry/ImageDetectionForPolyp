import cv2
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import albumentations as albu
from torch.utils.data import Dataset
from albumentations import pytorch as AT
import ttach as tta
from setting import *

# os.chdir("./code")


def resizeImg(img,crop_size=crop_size):

    return cv2.resize(img, crop_size, interpolation=cv2.INTER_AREA)

def padding(mask,input_size=input_size,crop_size=crop_size):
    L = (input_size[0]-crop_size[0])//2
    R = (input_size[0]-crop_size[0])-L
    T = (input_size[1]-crop_size[1])//2
    D = (input_size[1]-crop_size[1])-T

    return np.pad(mask,((L,R),(T,D)),constant_values=0)

def make_mask(df: pd.DataFrame, image_name: str = 'img.jpg', shape: tuple = (512, 512)):
    """
    Create mask based on df, image name and shape.
    :return list of mask(ndarray)
    """

    masks = df.loc[df['Images'] == image_name, 'Labels'].values
    rvt = np.zeros((shape[0], shape[1], 1))
    for idx, label in enumerate(masks):
        if label is not np.nan and label != '':
            label = label.split(';')
            img = np.zeros(shape)
            for mask in label:
                x,y,w,h = map(int,mask.split(' '))
                cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
            rvt[:,:,0]= img/255.
    return rvt

    # return masks


def to_tensor(x, **kwargs):
    """
    Convert image or mask.
    """
    # x = x.reshape((x.shape[0],x.shape[1],1))
    return x.transpose(2, 0, 1).astype('float32')
    # return x.astype('float32')

def visualize(image, mask, original_image=None, original_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    class_dict = {0: 'polyp',}

    if original_image is None and original_mask is None:
        f, ax = plt.subplots(1, 5, figsize=(24, 24))

        ax[0].imshow(image)
        for i in range(4):
            ax[i + 1].imshow(mask[:, :, i])
            ax[i + 1].set_title(f'Mask {class_dict[i]}', fontsize=fontsize)
    else:
        f, ax = plt.subplots(2, 2, figsize=(24, 12))

        ax[0, 0].imshow(original_image)
        ax[0, 0].set_title('Original image', fontsize=fontsize)

        for i in range(1):
            ax[0, i + 1].imshow(original_mask[:, :, i])
            ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

        ax[1, 0].imshow(image)
        ax[1, 0].set_title('Transformed image', fontsize=fontsize)

        for i in range(1):
            ax[1, i + 1].imshow(mask[:, :, i])
            ax[1, i + 1].set_title(f'Transformed mask {class_dict[i]}', fontsize=fontsize)


def tta_process(model):
    # transforms = tta.Compose(
    #     [
    #         # tta.HorizontalFlip(),
    #         # tta.V
    #         # tta.Rotate90(angles=[0, 180]),
    #         # tta.Scale(scales=[1, 2, 4]),
    #         # tta.Multiply(factors=[0.9, 1, 1.1]),        
    #     ]
    # )

    tta_model = tta.SegmentationTTAWrapper(model,tta.aliases.vlip_transform())
    return tta_model


def adapt_equalhist(image):
    """adaptive histogram equalization

    Arguments:
        image {np.array} -- image

    Returns:
        np.array -- image
    """
    clahe=cv2.createCLAHE(clipLimit=2,tileGridSize=(8,8))
    dst=clahe.apply(image)
    return dst

def visualize_with_raw(image, mask, original_image=None, original_mask=None, raw_image=None, raw_mask=None):
    """
    Plot image and masks.
    If two pairs of images and masks are passes, show both.
    """
    fontsize = 14
    # class_dict = {0: 'Fish', 1: 'Flower', 2: 'Gravel', 3: 'Sugar'}
    class_dict = {0: 'Polyp'}

    # f, ax = plt.subplots(3, 5, figsize=(24, 12))
    f, ax = plt.subplots(2, 2, figsize=(24, 12))

    ax[0, 0].imshow(original_image)
    ax[0, 0].set_title('Original image', fontsize=fontsize)

    for i in range(1):
        ax[0, i + 1].imshow(original_mask[:, :, i],cmap='gray')
        ax[0, i + 1].set_title(f'Original mask {class_dict[i]}', fontsize=fontsize)

    # ax[1, 0].imshow(raw_image)
    # ax[1, 0].set_title('raw Original image', fontsize=fontsize)

    # for i in range(1):
    #     ax[1, i + 1].imshow(raw_mask[:, :, i],cmap='gray')
    #     ax[1, i + 1].set_title(f'Raw predicted mask {class_dict[i]}', fontsize=fontsize)

    ax[1, 0].imshow(image)
    ax[1, 0].set_title('Transformed image', fontsize=fontsize)

    for i in range(1):
        ax[1, i + 1].imshow(mask[:, :, i],cmap='gray')
        ax[1, i + 1].set_title(f'Predicted mask with processing {class_dict[i]}', fontsize=fontsize)




sigmoid = lambda x: 1 / (1 + np.exp(-x))


def post_process(probability, threshold, min_size):
    """
    Post processing of each predicted mask, components with lesser number of pixels
    than `min_size` are ignored
    """
  
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((512, 512), np.float32)

    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    predictions = padding(resizeImg(predictions))

    contours, hier = cv2.findContours(predictions.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    img = np.zeros(predictions.shape)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
    return img, num

def predict_process(probability, threshold, min_size,shape=(512,512)):
    mask = cv2.threshold(probability, threshold, 1, cv2.THRESH_BINARY)[1]
    num_component, component = cv2.connectedComponents(mask.astype(np.uint8))
    predictions = np.zeros((512, 512), np.float32)

    num = 0
    for c in range(1, num_component):
        p = (component == c)
        if p.sum() > min_size:
            predictions[p] = 1
            num += 1
    # predictions = padding(resizeImg(predictions))

    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    ret=[]
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        ret.append((x/shape[0], y/shape[1], w/shape[0], h/shape[1]))
    return ret


def dice(img1, img2):
    """
    calculate correction
    :param img1:
    :param img2:
    :return: rate of dice
    """
    img1 = np.asarray(img1).astype(np.bool)
    img2 = np.asarray(img2).astype(np.bool)

    intersection = np.logical_and(img1, img2)

    return 2. * intersection.sum() / (img1.sum() + img2.sum())


# post_process_fn
def draw_convex_hull(mask, mode='rect'):
    """
    change contours's shape
    :param img1:
    :param img2:
    :return: rate of dice
    """
    img = np.zeros(mask.shape)
    contours, hier = cv2.findContours(mask.astype(np.uint8), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    for c in contours:
        if mode == 'rect':  # simple rectangle
            x, y, w, h = cv2.boundingRect(c)
            cv2.rectangle(img, (x, y), (x + w, y + h), (255, 255, 255), -1)
        elif mode == 'convex':  # minimum convex hull
            hull = cv2.convexHull(c)
            cv2.drawContours(img, [hull], 0, (255, 255, 255), -1)
        elif mode == 'approx':
            epsilon = 0.02 * cv2.arcLength(c, True)
            approx = cv2.approxPolyDP(c, epsilon, True)
            cv2.drawContours(img, [approx], 0, (255, 255, 255), -1)
        else:  # minimum area rectangle
            rect = cv2.minAreaRect(c)
            box = cv2.boxPoints(rect)
            box = np.int0(box)
            cv2.drawContours(img, [box], 0, (255, 255, 255), -1)
    return img / 255.

## polyp's dataset
class PolypDataset(Dataset):
    def __init__(self, df: pd.DataFrame = None, datatype: str = 'train', img_ids: np.array = None,
                 transforms = albu.Compose([albu.HorizontalFlip(),AT.ToTensor()]),
                preprocessing=None):
        self.df = df
        self.test=False
        if datatype == 'train':
            self.data_folder = f"{input_path}/train_images"
        elif datatype == 'test':
            self.data_folder = f"{input_path}/test_images"
            self.test=True
        elif datatype == 'valid':
            self.data_folder = f"{input_path}/valid_images"
            # self.data_folder = f"{input_path}/train_images"
        self.img_ids = img_ids
        self.transforms = transforms
        self.preprocessing = preprocessing

    def __getitem__(self, idx):
        image_name = self.img_ids[idx]
        mask=[]
        if self.test == False:
            mask = make_mask(self.df, image_name)
        else:
             mask = np.zeros((512, 512, 1), dtype=np.float32)
        image_path = os.path.join(self.data_folder, image_name)
        img = cv2.imread(image_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # img = img.reshape((img.shape[0],img.shape[1],1))
        augmented = self.transforms(image=img, mask=mask)
        img = augmented['image']
        mask = augmented['mask']
        if self.preprocessing:
            preprocessed = self.preprocessing(image=img, mask=mask)
            img = preprocessed['image']
            mask = preprocessed['mask']
        return img, mask

    def __len__(self):
        return len(self.img_ids)

