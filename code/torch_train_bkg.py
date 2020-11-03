import os
# os.chdir(f"{os.getcwd()}/code")
from torch.utils.data import DataLoader
# from helper_function import *
from preprocess import *
import torch
from  torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from mail import send_to_me
import segmentation_models_pytorch as smp
# from helper_function import  *


# parser
import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--continue_train',default=False,help='continue train or not')
parser.add_argument('--cuda_divice',default='0',help='CUDA_VISIBLE_DEVICES')
parser.add_argument('--bs',default=8,help='batch size',type=int)
parser.add_argument('--epochs',default=25,help='epochs',type=int)
parser.add_argument('--logdir',default="../log/inceptionresnetv2",help='log dir,where model and log are')
parser.add_argument('--send',default=True,help='send to lixunchao or not')
parser.add_argument('--msg',default='',help='addtional msg when send to me')
parser.add_argument('--train_pth',default=f"{input_path}/train.csv",help='path of train info(csv)')
parser.add_argument('--encode_lr',default=1e-3,help='encoder learning rate')
parser.add_argument('--decode_lr',default=1e-2,help='decoder learning rate')
parser.add_argument('--early_stop_patience',default=5,help='early stop patience')
parser.add_argument('--encoder',default='inceptionresnetv2',help='choose a below:'
                                                        'vgg11, vgg13, vgg16, vgg19, vgg11bn, vgg13bn, vgg16bn, vgg19bn,'
                                                        'densenet121, densenet169, densenet201, densenet161, dpn68, dpn98, dpn131,'
                                                        'inceptionresnetv2,'
                                                        'resnet18, resnet34, resnet50, resnet101, resnet152,'
                                                        'resnext50_32x4d, resnext101_32x8d,'
                                                        'se_resnet50, se_resnet101, se_resnet152,'
                                                        'se_resnext50_32x4d, se_resnext101_32x4d,'
                                                        'senet154')
arg = parser.parse_args()



if __name__ == "__main__":

    # setting
    DEVICE = 'cuda'
    num_workers = 0
    plot = False
    os.environ["CUDA_VISIBLE_DEVICES"] = arg.cuda_divice
    # os.environ["CUDA_VISIBLE_DEVICES"] = '0,1'

    bs = arg.bs
    epochs = arg.epochs
    logdir = arg.logdir
    send = arg.send
    train_csv_pth = arg.train_pth
    encode_lr = arg.encode_lr
    decode_lr = arg.decode_lr
    continue_train = arg.continue_train
    patience = arg.early_stop_patience
    addtional = arg.msg + '\n'


    # model
    ENCODER = arg.encoder
    # ENCODER_WEIGHTS = 'imagenet'
    # model = smp.Unet(
    #     encoder_name=ENCODER,
    #     encoder_weights=ENCODER_WEIGHTS,
    #     classes=4,
    #     activation="sigmoid"
    # )
    # preprocess_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)
    model,preprocess_fn  = get_model(ENCODER=ENCODER)
    # Parallel Gpu
    model = model
    # dataloader
    train = pd.read_csv(train_csv_pth)
    train_ids, valid_ids, test_ids = get_train_data()
    train_dataset = PolypDataset(df=train, datatype='train', img_ids=train_ids,
                                 transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocess_fn))
    valid_dataset = PolypDataset(df=train, datatype='valid', img_ids=valid_ids,
                                 transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(preprocess_fn))
    train_loader = DataLoader(train_dataset, bs, shuffle=True, num_workers=num_workers)
    valid_loader = DataLoader(valid_dataset, bs, shuffle=False, num_workers=num_workers)
    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }
    if continue_train is True:
        model.load_state_dict(torch.load(f"{logdir}/checkpoints/best.pth"))
    # optimizer
    optimizer = torch.optim.Adam([
        {'params':model.decoder.parameters(),'lr':decode_lr},
        {'params':model.encoder.parameters(),'lr':encode_lr},

    ])
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)

    # train
    runner = SupervisedRunner(device=DEVICE)
    runner.train(
        model=model,

        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        loaders=loaders,
        callbacks=[DiceCallback(), EarlyStoppingCallback(patience=5, min_delta=0.001)],
        logdir=logdir,
        num_epochs=epochs,
        verbose=True
    )
    # send to me
    if send is True:
        # addtional = ""
        with open(f"{logdir}/log.txt", 'r') as f:
            for line in f:
                addtional = addtional + line
        send_to_me(ENCODER,additinal= addtional)


