import torch
import timm

from ..ofa.model_zoo import ofa_flops_595m_s
from ..tresnet import TResnetM, TResnetL
from ..swin import build_swin_transformer_model
from src_files.helper_functions.distributed import print_at_master



def load_checkpoint_swin_t(config, model):
    checkpoint = torch.load(config.MODEL.RESUME, map_location='cpu')
    msg = model.load_state_dict(checkpoint['model'], strict=False)
    logger.info(msg)
    max_accuracy = 0.0
    if not config.EVAL_MODE and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        config.defrost()
        config.TRAIN.START_EPOCH = checkpoint['epoch'] + 1
        config.freeze()
        if 'amp' in checkpoint and config.AMP_OPT_LEVEL != "O0" and checkpoint['config'].AMP_OPT_LEVEL != "O0":
            amp.load_state_dict(checkpoint['amp'])
        logger.info(f"=> loaded successfully '{config.MODEL.RESUME}' (epoch {checkpoint['epoch']})")
        if 'max_accuracy' in checkpoint:
            max_accuracy = checkpoint['max_accuracy']

    del checkpoint
    torch.cuda.empty_cache()
    return max_accuracy


def load_model_weights(model, model_path):
    state = torch.load(model_path, map_location='cpu')
    for key in model.state_dict():
        if 'num_batches_tracked' in key:
            continue
        p = model.state_dict()[key]
        if 'state_dict' in state:
            key_state = 'module.'+ key
            if key_state in state['state_dict']:
                ip = state['state_dict'][key_state]
                if p.shape == ip.shape:
                    # print("key = ", key, "inshape = ", p.shape, "ipshape = ", ip.shape)
                    p.data.copy_(ip.data)  # Copy the data of parameters
                else:
                    print_at_master(
                        'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
            else:
                print_at_master('could not load layer: {}, not in checkpoint'.format(key))
        else:
            if 'model' in state:
                if key in state['model']:
                    ip = state['model'][key]
                    if p.shape == ip.shape:
                        p.data.copy_(ip.data)  # Copy the data of parameters
                    else:
                        print_at_master(
                            'could not load layer: {}, mismatch shape {} ,{}'.format(key, (p.shape), (ip.shape)))
                else:
                    print_at_master('could not load layer: {}, not in checkpoint'.format(key))
    return model


def create_model(args):
    print_at_master('creating model {}...'.format(args.model_name))

    model_params = {'args': args, 'num_classes': args.num_classes}
    args = model_params['args']
    args.model_name = args.model_name.lower()

    if args.model_name == 'tresnet_m':
        model = TResnetM(model_params)
    elif args.model_name == 'tresnet_l':
        model = TResnetL(model_params)
    elif args.model_name == 'ofa_flops_595m_s':
        model = ofa_flops_595m_s(model_params)
    elif args.model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=False, num_classes=args.num_classes)
    elif args.model_name == 'vit_base_patch16_224': # notice - qkv_bias==False currently
        model_kwargs = dict(
            patch_size=16, embed_dim=768, depth=12, num_heads=12, representation_size=None, qkv_bias=False)
        model = timm.models.vision_transformer._create_vision_transformer('vit_base_patch16_224_in21k',
                                                                          pretrained=False,
                                                                          num_classes=args.num_classes, **model_kwargs)
    elif args.model_name == 'mobilenetv3_large_100':
        model = timm.create_model('mobilenetv3_large_100', pretrained=False, num_classes=args.num_classes)
    elif args.model_name == 'swin_t':
        # args.cfg = "/workspace/mnt/storage/zhaozhijian/model_saving/ImageNet21K/ckpt/swin_tiny_patch4_window7_224.yaml"
        model = build_swin_transformer_model(args)
    else:
        print("model: {} not found !!".format(args.model_name))
        exit(-1)

    if args.model_path:  # make sure to load pretrained ImageNet-1K model
        model = load_model_weights(model, args.model_path)
    print('done\n')

    return model
