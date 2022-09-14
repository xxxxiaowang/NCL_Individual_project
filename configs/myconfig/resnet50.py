# _base_ from resnet50_b32x8_imagenet
_base_ = [
    "../_base_/default_runtime.py"
]  # 继承_base_的default_runtime.py中的runtime_setting设置

#########################################################DATASET#########################################################
img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53],
    std=[58.395, 57.12, 57.375],
    to_rgb=True
)

train_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(224, 224)),
    # dict(type = 'RandomFlip', flip_prob = 0.5, direction='horizontal'),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="ToTensor", keys=["gt_label"]),
    dict(type="Collect", keys=["img", "gt_label"]),
]
test_pipeline = [
    dict(type="LoadImageFromFile"),
    dict(type="Resize", size=(224, 224)),
    dict(type="Normalize", **img_norm_cfg),
    dict(type="ImageToTensor", keys=["img"]),
    dict(type="Collect", keys=["img"]),
]
# 定义dataloader和数据集
dataset_type = "CustomDataset"  # customized dataloader
data = dict(
    samples_per_gpu=2, # batch size
    workers_per_gpu=1, # 多线程 num_works
    train=dict(
        type=dataset_type,
        data_prefix="train_dataset\\train",  # 训练集路径
        pipeline=train_pipeline,
    ),
    val=dict(
        type=dataset_type,
        data_prefix="train_dataset\\val", #验证集路径
        pipeline=test_pipeline,
    ),
    test=dict(
        type=dataset_type,
        data_prefix="dataset",
        pipeline=test_pipeline,
    ),
)
##########################################################MODEL##########################################################
# 定义使用的模型backbone, neck, head
model = dict(
    type="ImageClassifier",
    backbone=dict(
        # frozen_stages=3, 该参数表示你想冻结前几个 stages 的权重，ResNet 结构包括 stem+4 stage
        type="ResNeSt",
        depth=50,
        num_stages=4,# ResNet 系列包括 stem+ 4个 stage 输出
        out_indices=(3,),# 表示本模块输出的特征图索引，(0, 1, 2, 3)/(3,), 表示4个stage 输出都需要,其stride 为 (4,8,16,32)，channel 为 (256, 512, 1024, 2048)
        style="pytorch",
    ),
    neck=dict(type="GlobalAveragePooling"),
    head=dict(
        type="LinearClsHead",
        num_classes=2,
        in_channels=2048,
        loss=dict(type="CrossEntropyLoss", loss_weight=1.0),
        topk=(1, 5),
    ),
)

########################################################SCHEDULE#########################################################
# paramwise configuration
optimizer = dict(type="SGD", lr=1e-3, momentum=1, weight_decay=1e-4)  # 优化器，学习率，动量
"""
    #范围参数
    paramwise_cfg = dict(
        custom_keys = {
            'backbone.cls_token':dict(decay_mult = 0),
            '...':dict(lr_mult = 0.9)
        }
    )
    or
    paramwise_cfg = dict(
        bias_lr_mult = 0.9,
        bias_decay_mult = 0.9,
        norm_decay_mult = 0.9,
        dcn_offset_lr_mult = 0.9,
        dwconv_decay_mult = 0.9
    )
"""
# gradient configuration #梯度设置
optimizer_config = dict(grad_clip=None)
"""
#梯度裁剪
#avoiding gradient explosion
optimizer_config = dict(
    grad_clip = dict(
        max_norm = 35, 
        norm_type = 2
    ),
    _delete_=True,
    type='OptimizerHook'
)
or
#梯度累积
#avoiding mini batch size
optimizer_config = dict(
    type = "GradientCumulativeOptimizerHook",
    cummulative_iters = 4
)
"""
# learning_rate and momentum configuration #学习率和动量参数，如周期性、衰减等

"""
#学习率衰减
lr_config = dict(policy='step', gamma=0.98, step=1)
or
lr_config = dict(policy='step',         
                 step=[30, 60, 90],
                 _delete_=True)
or
高斯学习率衰减
lr_config = dict(policy='CosineAnnealing', min_lr=1e-4, _delete_=True)
or
"""
# 周期性学习率+周期性动量
lr_config = dict(
    policy="cyclic",
    target_ratio=(10, 1e-4),  # max and min ratio of lr/lr_init
    cyclic_times=150,  # number of cycles during training
    step_ratio_up=0.5,  # the ratio of the increasing process of LR in the total cycle.
)
momentum_config = dict(
    policy="cyclic",
    target_ratio=(0.85, 0.95),  # min and max ratio of momentum/momentum_init
    cyclic_times=150,  # number of cycles during training
    step_ratio_up=0.5  # the ratio of the increasing process of LR in the total cycle.
    # _delete_=True
)

# epoch configuration #训练次数，如iterations和epochs
runner = dict(
    type="EpochBasedRunner",  # type of runner to use (i.e. IterBasedRunner or EpochBasedRunner)
    max_epochs=150,
)  # runner that runs the workflow in total max_epochs. For IterBasedRunner use `max_iters`

#####################################################RUNTIME_SETTING#####################################################
checkpoint_config = dict(interval=10, max_keep_ckpts=50)  # 每隔10次保存权重

log_config = dict(
    interval=50, hooks=[dict(type="TextLoggerHook"), dict(type="TensorboardLoggerHook")]
)

evaluation = dict(  # 计算准确率
    interval=1,
    metric=["accuracy", "precision", "recall"],
    metric_options={"topk": (1,)},
    save_best="auto",
    start=1,
)


workflow = [("train", 1), ("val", 1)]  # 每进行1次训练后进行1次验证
# workflow = [('test', 1)]
work_dir = "configs/org/work_dirs/resnet50_org"  # Path to save results
