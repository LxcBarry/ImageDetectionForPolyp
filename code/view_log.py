# view_log 只能在inference.py运行前使用，一旦被inference.py运行过该模型，它的训练信息会丢失
# 在训练信息查看.ipynb中有对应的模型训练后的log信息
# 也可以使用tensorboard实时查看训练信息
# 运行 tensorboard --logdir='log/{对应的模型名字}'

from catalyst.dl import utils
from setting import *
from configparser import ConfigParser

def view_train(model_name="inceptionresnetv2_of_extract"):
    cf = ConfigParser()
    logdir = f'../log/{model_name}'
    plot = True
    if plot == True:
        utils.plot_metrics(
            logdir=logdir,
            metrics=["loss","dice","lr","_bace/lr"]
        )
