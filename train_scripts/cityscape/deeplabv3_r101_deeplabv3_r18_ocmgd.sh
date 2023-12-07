CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-fd 0.000002 \
    --lambda-ctr 0.005 \
    --data data/cityscapes/ \
    --rand_mask False \
    --enhance_projector True \
    --save-dir work_dirs/deeplab \
    --log-dir log_dirs/deeplab \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth