"""
1. get train data
2. get model
3. preprocess label
4. preprocess images
"""
#%%
import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from PIL import Image
import matplotlib.pyplot as plt
import albumentations as albu
from albumentations import pytorch as AT
import segmentation_models_pytorch as smp
from setting import *
from helper_function import *

plot = False
#%%
##
def get_training_augmentation():
    """train augmentation
    in this function ,there is about 30 percents unchangeed

    Returns:
        function -- augmentation function
    """
    train_transform = [
        albu.CenterCrop(crop_size[0],crop_size[1]),
        albu.GaussianBlur(blur_limit=7,p=0.2),
        albu.GaussNoise(p=0.2),
        albu.VerticalFlip(p=0.2),
        albu.HorizontalFlip(p=0.2),
        albu.Rotate(p=0.2),
        albu.Resize(input_size[0],input_size[1])
    ]
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # albu.GridDistortion(p=1),
        # albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),
    return albu.Compose(train_transform)

##
def get_validation_augmentation():
    """Add paddings to make image shape divisible by 32"""
    test_transform = [
        albu.CenterCrop(480,480),
        albu.Resize(input_size[0],input_size[1])
    ]
    return albu.Compose(test_transform)

##






def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)
def get_preprocessing(preprocessing_fn):
    """Construct preprocessing transform

    Args:
        preprocessing_fn (callbale): data normalization function
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose

    """

    _transform = [
        albu.Lambda(image=preprocessing_fn),
        albu.Lambda(image=to_tensor, mask=to_tensor),
    ]
    return albu.Compose(_transform)

def warp_extractFeatures(Hurange=(80,140)):
    """warpper of function extractFeatures

    Keyword Arguments:
        Hurange {tuple} -- pixels range (default: {(80,140)})

    Returns:
        function -- wapper function
    """
    H=Hurange[1]
    L=Hurange[0]
    def warpper0(func):
        def warpper(*args,**kwargs):
            args[0][args[0] <= L] = 0
            args[0][args[0] >= H] = 0
            return func(*args,**kwargs)
        return warpper
    return warpper0

@warp_extractFeatures(Hurange=Hurange)
def transferImg(image="test\IM_0000.jpg",cols=input_size[0],rows=input_size[1]):
    """prepocessing image,most important function hear!!!

    Arguments:
        image {np.array} -- original images
        mask {np.array} -- mask

    Returns:
        transfered images
    """
    image = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    dst = np.zeros(image.shape)
    dst = cv2.equalizeHist(image)
    # dst = adapt_equalhist(image)
    dst = np.array([dst,dst,dst]).transpose(1,2,0)
    # dst = dst.reshape((dst.shape[0],dst.shape[1],1))
    return dst

# data
mask_tag=['polyp']
def get_train_data():
    """get train and test data,and create valid data mean while

    Returns:
        tuple -- file names of train imgages and valid images and test images
    """
    train = pd.read_csv(f'{input_path}/train.csv')
    train_ids=train['Images'].values
    valids = pd.read_csv(f'{input_path}/valids.csv')
    valid_ids = valids['Images'].values
    return train_ids,valid_ids

def get_test_data():
    """get test data

    Returns:
        list -- file names of test_ids
    """
    sub = pd.read_csv(f"{input_path}/test.csv")
    test_ids = sub['Images'].drop_duplicates().values
    return test_ids

# model
def get_model(ENCODER = 'resnet50',ENCODER_WEIGHTS = 'imagenet'):
    """get the unet model

    Keyword Arguments:
        ENCODER {str} -- unet's encoder (default: {'resnet50'})
        ENCODER_WEIGHTS {str} -- pretrain weight (default: {'imagenet'})

    Returns:
        tuple -- unet model and it's preprocess_function
    """
    model = smp.Unet(
        encoder_name=ENCODER,
        encoder_weights=ENCODER_WEIGHTS,
        classes=1, #只有一个类别
        # in_channels=1,
        # encoder_depth=5
        # activation= 'sigmoid'
    )
    preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    return model,preprocess_fn

def encode_squre(center_c,center_r,width,high):
    """encode the labels

    Arguments:
        center_c {str} -- column of center
        center_r {str} -- row of center
        width {str} -- width
        high {str} -- width

    Returns:
        str -- squre encode to int
    """
    center_r = int(pic_high*center_r)
    center_c = int(pic_width*center_c)
    width = int(width*pic_width)
    high = int(high*pic_high)
    ltc = center_c-width//2
    ltr = center_r-high//2
    code = [ltc,ltr,width,high]
    return (" ".join(map(str,code)))
def gather_data(datatype='train'):
    """create data list(as a csv file) for train and test

    Keyword Arguments:
        datatype {str} -- the type of gather (default: {'train'})
    """
    # 产生所有的数据列表
    df = pd.DataFrame()
    
    if datatype == 'train':
        files = list(os.listdir(f'{input_path}/train_images'))
        df['Images']=[x for x in files]

        labels = list(os.listdir(f'{input_path}/labels'))
        encode_list = []
        for f in files:
            if f[:-4]+'.txt' in labels:
                with open(f'{input_path}/labels/{f[:-4]}.txt','r') as F:
                    tmp_code = []
                    for line in F:
                        t,center_c,center_r,width,high = list(map(float,line.strip('\n').split(" ")))
                        x = center_c - width // 2
                        y = center_r - high // 2
                        tmp_code.append(encode_squre(center_c,center_r,width,high))

                    encode_list.append(";".join(tmp_code))
            else:
                encode_list.append("")
        df['Labels']=encode_list
        df.to_csv(f'{input_path}/train.csv',index=False)
    elif datatype=='test':
        files = os.listdir(f'{input_path}/test_images')
        df['Images']=[x for x in files]
        df.to_csv(f'{input_path}/test.csv',index=False)
    elif datatype=='valid':
        files = list(os.listdir(f'{input_path}/valid_images'))
        df['Images']=[x for x in files]

        labels = list(os.listdir(f'{input_path}/labels'))
        encode_list = []
        for f in files:
            if f[:-4]+'.txt' in labels:
                with open(f'{input_path}/labels/{f[:-4]}.txt','r') as F:
                    tmp_code = []
                    for line in F:
                        t,center_c,center_r,width,high = list(map(float,line.strip('\n').split(" ")))
                        x = center_c - width // 2
                        y = center_r - high // 2
                        tmp_code.append(encode_squre(center_c,center_r,width,high))

                    encode_list.append(";".join(tmp_code))
            else:
                encode_list.append("")
        df['Labels']=encode_list
        df.to_csv(f'{input_path}/valids.csv',index=False)
if __name__ == "__main__":
    # gather_data('test')
    gather_data('train')
    gather_data('valid')
    # train = pd.read_csv(f"{input_path}/train.csv")

    if plot is True:
        n_train = len(os.listdir(f"{input_path}/train_images"))
        n_test = len(os.listdir(f"{input_path}/test_images"))
        print(f'There are {n_train} images in train dataset')
        print(f'There are {n_test} images in test dataset')

    if plot is True:
        fig = plt.figure(figsize=(64, 64))
        for j, im_id in enumerate(np.random.choice(train['Images'].unique(), 2)):
            for i, (idx, row) in enumerate(train.loc[train['Images'] == im_id].iterrows()):
                ax = fig.add_subplot(2, 2, j * 2  + i+1, xticks=[], yticks=[])
                im = Image.open(f"{input_path}/train_images/{row['Images']}")
                plt.imshow(im)
                mask_code = row['Labels']
                try: # label might not be there!
                    mask = make_mask(train,im_id)
                    mask = mask.reshape(512,512)
                except:
                    mask = np.zeros((512, 512))
                plt.imshow(mask, alpha=0.5, cmap='gray')
    
        plt.savefig('test.png')







# %%
