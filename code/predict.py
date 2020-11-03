#%%
import pandas as pd
from tqdm import tqdm
from catalyst.dl.runner import SupervisedRunner
from preprocess import get_model
from torch.utils.data import DataLoader
from preprocess import gather_data
from configparser import  ConfigParser
import json
import numpy as np
import cv2
import os
from showLabelImg import labelImg
from showLabelImg import make_mask as mk_mask
from helper_function import *
from preprocess import *
from setting import *
os.environ["CUDA_VISIBLE_DEVICES"] = CUDA_VISIBLE_DEVICES
# os.environ["CUDA_VISIBLE_DEVICES"] = "0 1 2 3"


#%%



model,preprocess_fn = get_model(ENCODER)
print(logdir)

checkpoint=torch.load(f'{logdir}/checkpoints/best.pth')
model.load_state_dict(checkpoint['model_state_dict'])
#model.eval()
#model = tta_process(model)

#%%
def predict(test=False,draw = False,labels_path='labels'):
    if not os.path.exists(f'{output_path}'):
        os.mkdir(f"{output_path}")
    os.popen(f"rm -f {output_path}/*")
    # cf = ConfigParser()
    # cf.read(f'../model_config/{model_name}.ini')
    # logdir = f'../log/{model_name}'

    runner = SupervisedRunner()
    # 遍历test_images下的文件夹，得到test.csv
    gather_data('test')
    sub = pd.read_csv(f'{input_path}/test.csv')
    test_ids = test_ids = sub['Images'].drop_duplicates().values
    test_dataset = PolypDataset(df=sub,datatype='test',img_ids=test_ids,
                                transforms=get_validation_augmentation(),preprocessing=get_preprocessing(transferImg))
    test_loader = DataLoader(test_dataset,batch_size=bs,shuffle=False)
    loader = {'vaild':test_loader}
    image_id = 0
    preds = runner.predict_loader(model=model,
                          loader=test_loader,
                          resume=None,
                          #resume=f'{logdir}/checkpoints/best.pth',
                          verbose=True)
    result = []
    for i, output in enumerate(tqdm(preds)):
        # img, mask = batch
        for j, prob in enumerate(output):
            if prob.shape != (512, 512):
                prob = cv2.resize(prob, dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
            
            # crop_size -> input size
            # predict and get txt files
            contours=predict_process(sigmoid(prob), class_params[0][0],class_params[0][1])
            if len(contours) > 0:
                tmp_code = []
                with open(f'{output_path}/{test_ids[image_id][:-4]}.txt','w') as f:
                    for x,y,w,h in contours:
                        w = w * 0.85
                        h = h * 0.85
                        f.write(f"0 {x+0.5*w} {y+0.5*h} {w} {h}\n")
                        tmp_code.append(' '.join(map(str,[int(x*input_size[0]),int(y*input_size[1]),int(w*input_size[0]),int(h*input_size[1])])))
                result.append(';'.join(tmp_code))
            else:
                # result.append('')
                result.append(np.nan)
            image_id += 1
    sub['Labels'] = result
    sub.to_csv(cf['test']['sub'],index=False)

    if test is True:
        label_list = os.listdir(f"{input_path}/{labels_path}")
        cal = 0
        labels_num = 0
        pred_num = 0
        # recall precision f1----
        for i,x in enumerate(result):
            if x != np.nan and test_ids[i][:-4]+".txt" in label_list:
                cal += 1
            if test_ids[i][:-4]+".txt" in label_list:
                labels_num += 1
            if x != np.nan:
                pred_num += 1
        recall_score = cal / labels_num
        precision_score = cal /pred_num
        f1_score = 2*recall_score*precision_score/(recall_score+precision_score)
        print(
            f"recall_score:{recall_score}",
            f"precision_score={precision_score}",
            f"f1_score={f1_score}",
            sep="\n"
        )
        # dice
        d = []
        for i,x in enumerate(result):
            pred = make_mask(sub,f"{test_ids[i]}").astype(np.int8)
            valid = np.zeros(input_size)
            if test_ids[i][:-4]+".txt" in label_list:
                valid = mk_mask(file=f"{input_path}/{labels_path}/{test_ids[i][:-4]}.txt",mode='mask').astype(np.int8)
            if (pred.sum() == 0) & (valid.sum() == 0):
                d.append(1)
                # pass
            else:
                d.append(dice(pred,valid))
        d = np.array(d)
        print(
            "---------------------------",
            f"dice:{d.mean()}",
            sep = '\n'
        )
        if draw is True:
            labelImg(test_dir=f"{input_path}/test_images",label_dir=f"{output_path}")
            if(os.path.exists(f"{input_path}/labels")):
                labelImg(test_dir="test_img_output",label_dir=f"{input_path}/labels",output_dir=f"{output_path}/draw_output",color=(0,0,255))
            os.popen("rm -rf test_img_output")
#%%
predict(test=True,draw=True)

# %%
