#%%
import albumentations as albu
from albumentations import pytorch as AT
import os
import cv2
import matplotlib.pyplot as plt
from showLabelImg import *
#%%
input_size=(512,512)

def get_training_augmentation():
    train_transform = [
        albu.CenterCrop(480,480),
        albu.GaussianBlur(blur_limit=7,p=0.2),
        albu.GaussNoise(p=0.5),
        albu.VerticalFlip(p=0.5),
        albu.HorizontalFlip(p=0.5),
        albu.Resize(input_size[0],input_size[1])
        # albu.ShiftScaleRotate(scale_limit=0.5, rotate_limit=0, shift_limit=0.1, p=0.5, border_mode=0),
        # albu.GridDistortion(p=1),
        # albu.OpticalDistortion(p=1, distort_limit=2, shift_limit=0.5),

    ]
    return albu.Compose(train_transform)

#%%
test_dir = "one_plues_two"
out_dir = "augmentation_out"
label_dir = "labels_train"
if not os.path.exists(out_dir):
    os.mkdir(out_dir)

files = os.listdir(test_dir)

transform = get_training_augmentation()
for file in files:
    img = cv2.imread(f"{test_dir}/{file}")
    mask = make_mask(img,f"{label_dir}/{file[:-4]}.txt",mode='mask')
    augmented = transform(image=img,mask = mask)
    img = augmented['image']
    mask = augmented['mask'].astype(np.uint8)
    plt.imshow(img)
    plt.imshow(mask[:,:,0],cmap='gray',alpha=0.5)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"{out_dir}/{file}",cv2.addWeighted(img,0.7,mask,0.3,0))
    plt.show()

#%%