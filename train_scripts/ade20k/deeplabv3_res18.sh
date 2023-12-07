CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-kd 0.5 \
    --lambda-fd 0.000001 \
    --lambda-ctr 0.04 \
    --dataset ade20k \
    --data data/ade20k \
    --batch-size 16 \
    --workers 8 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 40000 \
    --save-dir work_dir/ade20k/ \
    --log-dir log_dir/ade20k/ \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_ade20k_best_model.pth  \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth



