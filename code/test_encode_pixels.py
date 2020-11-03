# 测试编码效果
import os
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from preprocess import *
path = '../input'
train_csv = f"{input_path}/train.csv"
train = pd.read_csv(train_csv)
im_id = ['IM_0011.jpg','IM_0012.jpg','IM_0013.jpg']

fig = plt.figure()
for i, (idx, row) in enumerate(train.loc[train['Images'] == im_id].iterrows()):
    ax = fig.add_subplot(3, 2, i*2+1, xticks=[], yticks=[])
    im = Image.open(f"{input_path}/train_images/{row['Images']}")
    plt.imshow(im)
    mask_code = row['Labels']
    try: # label might not be there!
        mask = make_mask(train,im_id)
        mask = mask.reshape(512,512)
    except:
        mask = np.zeros((512, 512))
    plt.imshow(mask, alpha=0.5, cmap='gray')
    ax = fig.add_subplot(3, 2, i*2+1, xticks=[], yticks=[])
    plt.imshow(mask, cmap='gray')

plt.savefig('test.png')