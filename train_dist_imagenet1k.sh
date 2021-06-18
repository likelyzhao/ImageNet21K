
python train_single_label_dist_imagenet1k.py \
    --batch_size=96 \
    --data_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ImageNet-pytorch \
    --model_name=swin_t \
    --model_path=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/model_best.pth.tar \
    --world-size 1 \
    --rank 0 \
    --cfg=/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ckpt/swin_tiny_patch4_window7_224.yaml \
    --multiprocessing-distributed \
    --dist-url "tcp://localhost:10001" \
    --epochs=60 


