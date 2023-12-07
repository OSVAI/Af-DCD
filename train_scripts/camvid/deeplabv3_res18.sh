CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-fd 0.000008 \
    --lambda-ctr 0.01 \
    --dataset camvid \
    --crop-size 360 360 \
    --data data/CamVid/ \
    --save-dir work_dir/camvid \
    --log-dir log_dir/camvid \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_camvid_best_model.pth  \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth



# CUDA_VISIBLE_DEVICES=0,1 \
#   python -m torch.distributed.launch --nproc_per_node=2 \
#   eval.py \
#   --model deeplabv3 \
#   --backbone resnet18 \
#   --dataset camvid \
#   --data data/CamVid/ \
#   --save-dir work_dir/camvid \
#   --pretrained work_dir/camvid/kd_deeplabv3_resnet18_camvid_best_model.pth