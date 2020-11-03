from torch.utils.data import DataLoader
import torch
from  torch.optim.lr_scheduler import ReduceLROnPlateau
from catalyst.dl.runner import SupervisedRunner
from catalyst.dl.callbacks import DiceCallback, EarlyStoppingCallback, InferCallback, CheckpointCallback
from mail import send_to_me
import segmentation_models_pytorch as smp
from preprocess import *
from setting import *
# 相关文档在这https://www.ctolib.com/qubvel-segmentation_models-pytorch.html


if __name__ == "__main__":

    os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
    model,preprocess_fn  = get_model(ENCODER=ENCODER)



    model = model
    # dataloader
    train = pd.read_csv(train_csv_pth)
    valid = pd.read_csv(valid_csv_pth)
    train_ids, valid_ids= get_train_data()
    # train_dataset = PolypDataset(df=train, datatype='train', img_ids=train_ids,
    #                              transforms=get_training_augmentation(),
    #                              preprocessing=get_preprocessing(preprocess_fn))
    train_dataset = PolypDataset(df=train, datatype='train', img_ids=train_ids,
                                 transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(transferImg))
    # valid_dataset = PolypDataset(df=train, datatype='valid', img_ids=valid_ids,
    #                              transforms=get_training_augmentation(),
    #                              preprocessing=get_preprocessing(preprocess_fn))
    valid_dataset = PolypDataset(df=valid, datatype='valid', img_ids=valid_ids,
                                 transforms=get_training_augmentation(),
                                 preprocessing=get_preprocessing(transferImg))
    train_loader = DataLoader(train_dataset, bs, shuffle=True,num_workers=0)
    valid_loader = DataLoader(valid_dataset, bs, shuffle=False,num_workers=0)
    loaders = {
        'train': train_loader,
        'valid': valid_loader
    }



    if continue_train is True:
        model.load_state_dict(torch.load(f"{logdir}/checkpoints/best.pth"))
    # optimizer
    # optimizer = torch.optim.Adam([
    #     {'params':model.decoder.parameters(),'lr':decode_lr},
    #     {'params':model.encoder.parameters(),'lr':encode_lr},

    # ])
    optimizer = optimizer_chooise([
        {'params':model.decoder.parameters(),'lr':decode_lr},
        {'params':model.encoder.parameters(),'lr':encode_lr},

    ])
    
    
    scheduler = ReduceLROnPlateau(optimizer, factor=0.15, patience=2)
    criterion = loss_fun
    criterion = smp.utils.losses.BCEDiceLoss(eps=1.)
    # criterion = smp.utils.losses.BCEDiceLoss(eps=1.)


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
    # torch.save(model,model_path)
    # send to me
    if send is True:
        # addtional = ""
        with open(f"{logdir}/log.txt", 'r') as f:
            for line in f:
                addtional = addtional + line
        send_to_me(ENCODER,additinal= addtional)






