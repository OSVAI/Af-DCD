CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplab_mobile \
    --teacher-backbone resnet101 \
    --student-backbone mobilenetv2 \
    --lambda-fd 0.000002 \
    --lambda-ctr 0.005 \
    --data data/cityscapes/ \
    --save-dir work_dirs/deeplabv3_r101_mbv2_r18_cityscape/ \
    --log-dir log_dirs/deeplabv3_r101_mbv2_r18_cityscape/ \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_citys_best_model.pth \
    --student-pretrained-base pretrained_ckpt/mobilenetv2-imagenet.pth