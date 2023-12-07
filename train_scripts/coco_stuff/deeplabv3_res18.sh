CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8  \
    train_ocmgd.py \
    --teacher-model deeplabv3 \
    --student-model deeplabv3 \
    --teacher-backbone resnet101 \
    --student-backbone resnet18 \
    --lambda-kd 0.5 \
    --lambda-fd 0.000005 \
    --lambda-ctr 0.005 \
    --dataset coco_stuff_164k \
    --data data/coco_stuff_164k/ \
    --batch-size 16 \
    --workers 8 \
    --crop-size 512 512 \
    --lr 0.02 \
    --max-iterations 80000 \
    --save-dir work_dir/cocostuff/ \
    --log-dir log_dir/cocostuff/ \
    --teacher-pretrained pretrained_ckpt/deeplabv3_resnet101_coco_stuff_164k_best_model.pth \
    --student-pretrained-base pretrained_ckpt/resnet18-imagenet.pth
