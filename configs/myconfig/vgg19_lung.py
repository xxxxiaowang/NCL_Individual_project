_base_ = ["../_base_/default_runtime.py"]
#默认运行设置，模型保存、显卡等配置

img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True
)#  图像归一化参数。对输入图片进行标准化处理的配置，减去mean，除以std，要将读取的（默认）bgr转为rgb排列

train_pipeline = [
    dict(type="LoadImageFromFile"),#首先读取数据
    dict(type="Resize", size=(224, 224)),#The image size was originally about 3000*3000，这里resize一下
    dict(type="Normalize", **img_norm_cfg),#用之前的img_norm_cfg参数进行图像标准化
    dict(type="ImageToTensor", keys=["img"]),#image 转为 torch.Tensor
    dict(type="ToTensor", keys=["gt_label"]),#gt_label 转为 torch.Tensor
    dict(type="Collect", keys=["img", "gt_label"]),#表示可以被传入pipeline的的keys
]    # 数据增强data enhancement 相当于pytorch里的transformer
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(224, 224)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),#test时不传递gt_label
]
#以上两块都是数据流水线
# train : val = 7:3
# dataset_type = "DATALOADER" 
dataset_type = "CustomDataset" 
data = dict(
    samples_per_gpu=1, # （单个GPU的）batch size
    workers_per_gpu=1, #  (单个GPU的) 线程数 num_works
    train=dict(
        type=dataset_type,
        data_prefix=r"D:/mmclassification/train_dataset/train",  # 训练集路径 
        pipeline=train_pipeline, #数据集 需要经过的数据流水线
    ),
    val=dict(
        type=dataset_type,
        #data_prefix="train_dataset\\train",
        data_prefix="train_dataset\\val", #验证集路径
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_prefix=r"D:/mmclassification/train_dataset/train",
        pipeline=test_pipeline,
    ),
)


evaluation = dict(
    interval=1,#验证期间的间隔，单位一般为epoch或iter，取决于runner类型
    metric=["accuracy", "precision", "recall"], #验证期间使用的指标
    metric_options={"topk": (1,)},
    save_best="auto",
)


# model settings model部分分为backbone, neck, head以及其他配置。
model = dict(
    type="ImageClassifier",# model name 即分类器类型
    backbone=dict(type="VGG", depth=19, norm_cfg=dict(type="BN"), num_classes=2),# VGG是backnone（主干）网络类型；16层；norm layer的类型是BN；输出类别数是2类，与数据集一致
    neck=None,
    head=dict(
        type="ClsHead",# 对应上面的head类
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1,),
    ),
)

# optimizer 用于构建优化器的配置文件。支持 PyTorch 中的所有优化器，同时它们的参数与 PyTorch 里的优化器参数一致。
optimizer = dict(type="SGD", lr=0.0001, momentum=0.9, weight_decay=0.00001)#优化器类型是SGD；lr，momentum是SGD的一种加速超参；weight_decay是权重惩罚参数
optimizer_config = dict(grad_clip=None) #大多数方法不使用梯度限制（grad_clip）
# learning policy 学习率调整配置
lr_config = dict(policy="step", step=[30, 60, 90])#在epoch为30，60，90时lr进行衰减
runner = dict(type="EpochBasedRunner", max_epochs=100) # 将使用的runner的类别，如EpochBasedRunner或IterBasedRunner；Epochs 100


work_dir = "configs/org/work_dirs/vgg16_lung_20220812V19_org"
#用于保存当前实验的模型检查点和日志的目录文件地址 记得每次训练都要改一下地址

# workflow = [("train", 1), ("val", 1)] # 每进行1次训练后进行1次验证