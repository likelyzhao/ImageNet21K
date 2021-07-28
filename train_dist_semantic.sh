

python -u train_semantic_dist.py \
    --batch_size=96 \
    --lr 5e-5 \
    --data_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/imagenet11k \
    --model_name=swin_t \
    --model_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ckpt/swin_nano_patch4_window7_224.pth \
    --world-size 1 \
    --rank 0 \
    --cfg=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ckpt/swin_nano_patch4_window7_224.yaml \
    --multiprocessing-distributed \
    --dist-url "tcp://localhost:10001" \
    --epochs=80 \
    --tree_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ckpt/imagenet21k_miil_tree.pth




