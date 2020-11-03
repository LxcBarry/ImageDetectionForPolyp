#%%
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
import random
from setting import input_size
Hurange=(80,120)

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

def transferImgs(test_dir = "test",out_dir = "extract_out",show=False):
    """use equalizeHist to transfer images

    Keyword Arguments:
        test_dir {str} -- input dir (default: {"test"})
        out_dir {str} -- output dir (default: {"extract_out"})
    """ 
    files = os.listdir(test_dir)
    for file in files:
        img = cv2.imread(f"{test_dir}/{file}")
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dst = cv2.equalizeHist(gray)
        if show is True:
            plt.imshow(dst,cmap='gray')
            plt.show()
        cv2.imwrite(f"{out_dir}/{file}",dst)


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
    dst = np.array([dst,dst,dst]).transpose(1,2,0)
    return dst


def extractFeatures(HuRange=(80,140),test_dir = "test",out_dir = "extract_out",show=False):
    """extract a range of pixels,create new images

    Keyword Arguments:
        HuRange {tuple} -- pixels range (default: {(80,140)})
        test_dir {str} -- input dir (default: {"test"})
        out_dir {str} -- output dir (default: {"extract_out"})
    """
    files = os.listdir(test_dir)
    # H = lambda x:(HuRange[1]+1024)/x*255.
    # L = lambda x:(HuRange[0]+1024)/x*255.
    H=HuRange[1]
    L=HuRange[0]
    for file in files:
        img = cv2.imread(f"{test_dir}/{file}")
        img = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        # tmp = np.zeros(img.shape)
        # tmp[img <= H & img >= L] = img[img <= H & img >= L]
        img[img >= H]=0
        # img[(img < H) & (img > L)] = img[(img < H) & (img > L)] / (H-L) * 255
        # range_x = img[img<255].max()-img[img>0].min()
        # range_x = range_x.astype(np.float)
        # range_x = range_x / 255. * 4095
        # img[img >= H(range_x)]=0
        # plt.show()
        # img[img <= L(range_x)]=0
        img[img <= L]=0
        # cv2.imshow("",img)
        # cv2.waitKey(0)
        # cv2.destroyAllWindows("")
        if show is True:
            plt.imshow(img,cmap='gray')
            plt.show()
        cv2.imwrite(f"{out_dir}/{file}",img)

        

def testDistribution(test_dir = "test"):
    """test the pixels distribution

    Keyword Arguments:
        test_dir {str} -- input dir (default: {"test"})
    """
    files = os.listdir(test_dir)

    choises = random.sample(files,3)
    collect = []
    for file in choises:
        img = cv2.imread(f"{test_dir}/{file}")
        collect.extend(img.reshape(-1).tolist())

    plt.hist(collect,bins=20,histtype='bar',rwidth=0.8)
    plt.xticks(np.arange(0,200,20))
    plt.title("pixels distribution")
    plt.show()




# if __name__ == "__main__":
    # transferImgs()
    # extractFeatures()
    # testDistribution("one_plues_two")
    # transferImgs(test_dir='extract_out',out_dir='one_plus_two',show=True)
#%%
img = cv2.imread(f"../input/train_images/IM_0000.jpg")
gets = transferImg(img)
# cv2.imshow("",gets)
plt.imshow(gets,cmap='gray')
plt.show()
# cv2.waitKey(0)
# cv2.destroyAllWindows("")



# %%
