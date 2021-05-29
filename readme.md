# ai2020


# 用法
```sh
# 1. 
conda activate satellite_lxc

# 2.
cd /home/gfl/LxcTest/ai2020/kaggle_satellite/code

# 3. 按照自己的配置，做一份ini文件(在model_config)，要求命名唯一

# 4. 在train_config.py中修改model_name为刚才的名字

# 5. 运行训练代码
python torch_train.py

# 7. 训练完会发邮件，如果想在训练完发邮件提醒，在mail.py中添加自己账户

# 8. 打开观察文件"训练信息查看.ipynb",查看运行结果

# 9. 调优，运行inference.py，在inference_config.py中修改想要调优的模型名字，修改后运行以下命令
python inference.py

# 10. 测试，修改predict.py中的model_name可以选择测试的模型
python predict.py

# 11. 查看加上mask后的预测图片，修改showLabelImg.py中的输入参数，即可查看
python showLabelImg.py
```

# 运行逻辑

conda环境是satellite_lxc
```
conda activate satellite_lxc
```

## 训练
目前预先配置好了几种模型。在model_config下，里面可以配置模型训练的参数和指定的gpu  
训练前，在code/train_cofig.py下设置训练的模型，只要填写模型名字即可  
训练后的模型在log下面  

目前训练好能用的模型：  
- resnet34:分割精度0.56

待训练的模型
- resnet50
- densenet121


## 预测
helper_function.py中有一个predict_process，可以将预测到的结果输出成txt，将图片放入input/test_images下(所有图片)，运行code/predict.py，会生成input/test.csv  
并且把预测结果存到output

<!-- ## 优化部分
这部分有点搞不明白，代码在inference.py里边，这份代码是选取log下面的某个模型(通过train_config.py配置)，选取其中的最优模型，调用infer函数，获取预测结果，我的思路是，如果输出的是概率，那么修改概率阈值和最小mask大小应该有不同效果，但是修改阈值没效果，修改mask大小有一点效果。但是效果不明显，目前resnet34的图片分割精度在0.5以上。我猜是因为模型预测出来的结果里面，数值分布在两个极端，将这个输出值再经过sigmoid后所得到的结果都是再0，1附近，这样使得最后的修改阈值所得到的结果差异不大，没有优化效果，所以我目前的想法是怎么修改模型让他输出的结果是一个正常的概率值，这样我就可以用概率阈值优化结果了。 -->

## 训练信息查看
运行view_log.py

## 模型获取
使用segmentation_models_pytorch的预训练模型，在preprocess.py下面有一个get_model，返回一个模型，还有输入到这个模型的预处理函数(我猜是做图片大小转换的)  
具体segmentation_models_pytorch的用法我当时看的不多，只知道设置好encoder和输出的类别有哪些就可以了。  
找到一个链接和这个代码库相关[链接](https://www.ctolib.com/qubvel-segmentation_models-pytorch.html#encoders)  

如果要新建一种模型的话，可以先在model_config下面复制一份前面的ini文件,改了encoder，运行的时候，该了train_config.py的encoder，再次运行torch_train就行   
如果这个模型不是对应是unet的解码模型的话，可能需要改改get_model里边的代码  

torch_train_bkg.py代码和torch_train.py差不多的，只是读取参数的位置不一样，其它没什么区别

## 数据集合获取
使用dataloader，在helper_function.py下面定义了以一个polypdataset,这个模型会自动地根据输入的的配置进行扩增

## 数据扩增
数据扩增作为一个转换函数输入到dataset中，在helper_function下面有两种扩增get_training_augmentation和get_valids_augmentation，后一个没扩增，前一个可以翻转等等操作，可以使用view_augmentation.py查看扩增效果  

## 图片
画图的话可以用img下面的图片，里边有几个训练后的预测结果的图片，也可以用view_augmentation.py获取扩增的图片






