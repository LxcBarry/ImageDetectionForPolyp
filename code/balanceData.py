#%%
"""count the label img and unlabel img"""
import numpy as np
import matplotlib.pyplot as plt
import os
import random
from setting import input_path,output_path

#%%
img_path = f"{input_path}/image_train"
label_path = f"{input_path}/labels"
test_img_path = f"{input_path}/test_images"
train_img_path = f"{input_path}/train_images"
valid_img_path = f"{input_path}/valid_images"
img_files = os.listdir(img_path)

label_imgs = [f for f in img_files if os.path.exists(f"{label_path}/{f[:-4]}.txt")]
unlabel_imgs = [f for f in img_files if not os.path.exists(f"{label_path}/{f[:-4]}.txt")]

plt.bar(['labeled','nonlabeled'],[len(label_imgs),len(unlabel_imgs)])
plt.title('data distribution')
plt.show()
# label:857, nonlabel:643

# %%
# 8:2(or other choise)split data,mixed with both label-images and nonlabel-images

# partial=(0.8,0.2)
# shuffle data
random.shuffle(label_imgs)
random.shuffle(unlabel_imgs)
train_imgs = label_imgs[:int(0.9*len(label_imgs))]+unlabel_imgs[:int(0.9*len(unlabel_imgs))]
# test_imgs = label_imgs[int(0.7*len(label_imgs)):int(0.9*len(label_imgs))]+unlabel_imgs[int(0.7*len(unlabel_imgs)):int(0.9*len(label_imgs))]
valid_imgs = label_imgs[int(0.9*len(label_imgs)):]+unlabel_imgs[int(0.9*len(unlabel_imgs)):]
# valid_imgs = label_imgs[int(0.9*len(label_imgs)):]+unlabel_imgs[int(0.9*len(unlabel_imgs)):]

# %%
if not os.path.exists(train_img_path):
    os.mkdir(train_img_path)
# if not os.path.exists(test_img_path):
#     os.mkdir(test_img_path)
if not os.path.exists(valid_img_path):
    os.mkdir(valid_img_path)

for file in train_imgs:
    os.popen(f'cp {img_path}/{file} {train_img_path}/{file}')
# for file in test_imgs:
#     os.popen(f'cp {img_path}/{file} {test_img_path}/{file}')
for file in valid_imgs:
    os.popen(f'cp {img_path}/{file} {valid_img_path}/{file}')

print(f"len train:{len(train_imgs)}|len valid:{len(valid_imgs)}")
# %%
