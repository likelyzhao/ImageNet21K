

python train_single_label_dist.py \
    --batch_size=64 \
    --data_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/imagenet11k \
    --model_name=swin_t \
    --model_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ckpt/swin_tiny_patch4_window7_224.pth \
    --world-size 1 \
    --rank 0 \
    --multiprocessing-distributed \
    --dist-url "tcp://localhost:10001" \
    --epochs=80 




