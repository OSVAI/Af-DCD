CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
    python -m torch.distributed.launch --nproc_per_node=8 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model psp \
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

# CUDA_VISIBLE_DEVICES=0 \
#   python -m torch.distributed.launch --nproc_per_node=1 \
#   eval.py \
#   --model psp \
#   --backbone resnet18 \
#   --dataset camvid \
#   --data [your dataset path]/CamVid/ \
#   --save-dir [your directory path to store checkpoint files] \
#   --pretrained [your pretrained model path]
