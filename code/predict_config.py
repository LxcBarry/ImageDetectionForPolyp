
#%%
from configparser import ConfigParser

# choose one model to train or predict
# 实验组，图片分层
# model_name = 'inceptionresnetv2_of_extract'
# model_name = 'resnet34_of_extract'
# model_name = 'resnet34_of_extract_Tw301_BCEDloss'
# model_name = 'resnet34_of_extract_w301_diceloss'
model_name = 'inceptionresnetv2_of_extract_w308'
ini_pth = f'../model_config/{model_name}.ini'
cf = ConfigParser()
cf.read(ini_pth)
CUDA_VISIBLE_DEVICES = cf['DEFAULT'].get('cuda_visible_devices')
DEVICE = cf['DEFAULT']['device']
ENCODER = cf['DEFAULT']['encoder']
Hurange = eval(cf['DEFAULT']['hurange'])
crop_size = eval(cf['DEFAULT']['crop_size'])
# logdir = cf['DEFAULT']['logdir']
logdir = f'../log/{model_name}'
bs = cf.getint('test', 'bs')
epochs = cf.getint('train', 'epochs')
# send = cf['train']['send']
send = cf.getboolean('train','send')
train_csv_pth = cf['train']['train_csv_pth']
valid_csv_pth = cf['train']['valid_csv_pth']
encode_lr = cf['train'].getfloat('encode_lr')
decode_lr = cf['train'].getfloat('decode_lr')
continue_train = cf.getboolean('train', 'continue_train')
patience = cf.getint('train', 'patience')
addtional = cf['train']['addtional']
class_params = eval(cf.get('test','class_params'))


# %%
