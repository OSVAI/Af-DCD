CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-fd 0.000001 \
    --lambda-ctr 0.005 \
    --dataset voc \
    --crop-size 512 512 \
    --data data/VOCAug/ \
    --save-dir work_dir/VOCAug/deeplabv3_res18 \
    --log-dir log_dir/VOCAug/deeplabv3_res18 \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_pascal_aug_best_model.pth \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth

