CUDA_VISIBLE_DEVICES=0,1 \
python -m torch.distributed.launch --nproc_per_node=2 save_embeddings.py \
    --model deeplab_mobile \
    --backbone mobilenetv2 \
    --dataset citys \
    --data ../data/cityscapes/ \
    --save-dir ./ \
    --pretrained /data2/anbang/fjw/kd_codebase/CIRKD/vis_ckpt/student.pth

python tsne_visual.py