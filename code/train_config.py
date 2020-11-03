
#%%
from configparser import ConfigParser
import segmentation_models_pytorch as smp
import torch
# choose one model to train or predict
# 实验组，图片分层
# model_name = 'inceptionresnetv2_of_extract'
#model_name = 'resnet34_of_extract'
# model_name = 'inference_no_pretrain_extract'
# 对照组，没有图片分层，全部都输入到模型
model_name = 'inceptionresnetv2_of_extract_w308'
#model_name = 'test1'
# model_name = 'inceptionresnetv2_of_noextract'
# model_name = 'resnet34_of_noextract'


# model_name = 'vgg16'
# model_name = 'inceptionresnetv2'
# model_name = 'densenet121'
# 正在训练
# ini_pth = f'../model_config/{model_name}.ini'
ini_pth = f'../model_config/{model_name}.ini'
cf = ConfigParser()
cf.read(ini_pth)
CUDA_VISIBLE_DEVICES = cf['DEFAULT'].get('cuda_visible_devices')
DEVICE = cf['DEFAULT']['device']
ENCODER = cf['DEFAULT']['encoder']
Hurange = eval(cf['DEFAULT']['Hurange'])
crop_size = eval(cf['DEFAULT']['crop_size'])
# logdir = cf['DEFAULT']['logdir']
logdir = f'../log/{model_name}'
loss_fun = eval(cf['train']['Loss'])
bs = cf.getint('train', 'bs')
epochs = cf.getint('train', 'epochs')
optimizer_chooise=eval(cf['train']['optimizer'])
# send = cf['train']['send']
send = cf.getboolean('train','send')
train_csv_pth = cf['train']['train_csv_pth']
valid_csv_pth = cf['train']['valid_csv_pth']
encode_lr = cf['train'].getfloat('encode_lr')
decode_lr = cf['train'].getfloat('decode_lr')
continue_train = cf.getboolean('train', 'continue_train')
patience = cf.getint('train', 'patience')
addtional = cf['train']['addtional']

# %%
