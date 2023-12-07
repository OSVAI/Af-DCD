CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 \
python -m torch.distributed.launch --nproc_per_node=8 eval.py \
    --model deeplabv3 \
    --backbone resnet18 \
    --dataset '[dataset name]' \
    --data '[dataset dir]' \
    --save-dir '[save dir]' \
    --pretrained '[checkpoint path]'