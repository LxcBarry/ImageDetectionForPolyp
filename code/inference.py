# 待修改文件
#%%
from preprocess import get_model
from tqdm import tqdm # roll of progress
import pandas as pd
import cv2
import seaborn as sns
import matplotlib.pyplot as plt

from inference_config import *
from helper_function import *
from preprocess import *
from setting import *
#-----------------------------------------------------------------------
#%%
runner = SupervisedRunner()
model,preprocess_fn = get_model(ENCODER)
valid = pd.read_csv(valid_csv_pth)
valid_ids = pd.read_csv(f"{input_path}/valids.csv")
valid_ids = valid_ids['Images'].values
valid_dataset = PolypDataset(df=valid, datatype='valid', img_ids=valid_ids,
                             transforms=get_validation_augmentation(),
                             preprocessing=get_preprocessing(transferImg))
valid_loader = DataLoader(valid_dataset, bs, shuffle=False,num_workers=0)

os.environ["CUDA_VISIBLE_DEVICES"]=CUDA_VISIBLE_DEVICES

#%%
loaders = {'infer':valid_loader}
runner.infer(
    model= model,
    loaders = loaders,
    callbacks=[
        CheckpointCallback(
            resume=f"{logdir}/checkpoints/best.pth"
        ),
        InferCallback()
    ],
    verbose=True
)

#%%
vaild_mask = []
probabilities = np.zeros((len(valid_dataset)*1,512,512))
count = 0
for i,(batch,output) in enumerate(tqdm(zip(valid_dataset,runner.callbacks[0].predictions["logits"]))):
    img,mask = batch
    for j,(m,out) in enumerate(zip(mask,output)):
        if m.shape != (512,512):
            m = cv2.resize(m,dsize=(512,512),interpolation=cv2.INTER_LINEAR)
        vaild_mask.append(m)

    for j, prob in enumerate(output):
        if prob.shape != (512,512):
            prob = cv2.resize(prob,dsize=(512,512),interpolation=cv2.INTER_LINEAR)

        probabilities[count,:,:] = prob #只有一个类别
        count += 1


#%%
class_params = {}

top = pd.DataFrame(index=['Polyp'],columns=['threshold', 'size', 'dice']) #一个类别
choises_size=[0,4*4,8*8,16*16,24*24]
thresh_holds = range(10,100,5)
for class_id in range(1):
    print(f"-- {mask_tag[class_id]} --")
    attempts = []
    for ts in tqdm(thresh_holds):
        ts /= 100
        for ms in choises_size:
            masks = []
            print(f"-------ms:{ms},ts:{ts}--------")
            for i in range(class_id,len(probabilities)):
                prob = probabilities[i]
                pred,num_pred = post_process(sigmoid(prob),ts,ms)
                # pred,num_pred = post_process(prob,ts,ms)
                # pred,num_pred = post_process(prob,ts,ms)
                masks.append(pred)

            d = []
            for i,j in zip(masks,vaild_mask):
                if (i.sum() == 0) & (j.sum() == 0):
                    d.append(1)
                    # pass
                else:
                    d.append(dice(i,j))

            print(np.mean(d))    
            attempts.append((ts,ms,np.mean(d)))

    attempts_df = pd.DataFrame(attempts, columns=['threshold', 'size', 'dice'])

    attempts_df = attempts_df.sort_values('dice', ascending=False)
    top.iloc[class_id]=attempts_df.iloc[0]

    print(attempts_df.head())
    best_threshold = attempts_df['threshold'].values[0]
    best_size = attempts_df['size'].values[0]

    class_params[class_id] = (best_threshold, best_size)
    fig = plt.figure()
    
    for size in choises_size:
        plt.plot(thresh_holds,attempts_df[attempts_df['size']==size]['dice'].values,label=str(size))
    # sns.lineplot(x = 'threshold',y='dice',hue='size',data=attempts_df)
    # plt.title(f'Threshold and min size vs dice for class {mask_tag[class_id]}')
    plt.legend()
    plt.show()
    plt.savefig(f"../img/{model_name}_{mask_tag[class_id]}.jpg")

attempts_df[:10].to_csv(f'../model_config/{model_name}_threshold.csv')



for i, (input, output) in enumerate(zip(valid_dataset, runner.callbacks[0].predictions["logits"])):
    image, mask = input
    image_vis = image.transpose(1, 2, 0)
    mask = mask.astype('uint8').transpose(1, 2, 0)
    pr_mask = np.zeros((512, 512, 1))
    for j in range(1):
        probability = cv2.resize(output.transpose(1, 2, 0)[:, :, j], dsize=(512, 512), interpolation=cv2.INTER_LINEAR)
        pr_mask[:, :, j], _ = post_process(sigmoid(probability), class_params[j][0], class_params[j][1])
    # pr_mask = (sigmoid(output) > best_threshold).astype('uint8').transpose(1, 2, 0)

    visualize_with_raw(image=image_vis/255, mask=pr_mask.astype(np.uint8), original_image=image_vis/255, original_mask=mask.astype(np.uint8))
    plt.show()
    plt.savefig(f"../img/visualize_{model_name}_{i}.jpg")

    if i >= 2:
        break


# save class params
with open(ini_pth,'w') as f:
    cf['test']["class_params"] = str(class_params)
    cf.write(f)

# %%
