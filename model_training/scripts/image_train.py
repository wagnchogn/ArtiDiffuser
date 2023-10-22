"""
Train a diffusion model on images.
"""

import argparse

from guided_diffusion import dist_util, logger
from guided_diffusion.image_datasets import load_data
from guided_diffusion.resample import create_named_schedule_sampler
from guided_diffusion.script_util import (
    model_and_diffusion_defaults,
    create_model_and_diffusion,
    args_to_dict,
    add_dict_to_argparser,
)
from guided_diffusion.train_util import TrainLoop
import thop
import torch
#from torchsummary import summary
def main():
    args = create_argparser().parse_args() #定义超参数

    dist_util.setup_dist()
    logger.configure(dir='/root/autodl-tmp/class_guided_artifusion_mse_mmoe/log/1008_ablution')

    logger.log("creating model and diffusion...")
    model, diffusion = create_model_and_diffusion(
        **args_to_dict(args, model_and_diffusion_defaults().keys())
    )

    model.to(dist_util.dev())
    '''
    input_tensor = torch.randn(5, 3, 256, 256).to(dist_util.dev())
    time_info = torch.tensor([5,5,5,5,5]).to(dist_util.dev())
    category_info = torch.tensor([3,1,2,0,4]).to(dist_util.dev())
    
    input_tensor = torch.randn(1, 3, 256, 256).to(dist_util.dev())
    time_info = torch.tensor([5]).to(dist_util.dev())
    category_info = torch.tensor([3]).to(dist_util.dev())
    flops, params = thop.profile(model, inputs=(input_tensor, time_info, category_info))
    print(flops)
    print(params)
    '''
    schedule_sampler = create_named_schedule_sampler(args.schedule_sampler, diffusion)#均匀采样器还是重要性采样器

    logger.log("creating data loader...")
    data = load_data(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        image_size=args.image_size,
        class_cond=args.class_cond,
    )

    logger.log("training...")
    # TrainLoop是一个类，run_loop是一个方法
    TrainLoop(
        model=model,
        diffusion=diffusion,
        data=data,
        batch_size=args.batch_size,
        microbatch=args.microbatch,
        lr=args.lr,
        ema_rate=args.ema_rate,
        log_interval=args.log_interval,
        save_interval=args.save_interval,
        resume_checkpoint=args.resume_checkpoint,
        use_fp16=args.use_fp16,
        fp16_scale_growth=args.fp16_scale_growth,
        schedule_sampler=schedule_sampler,
        weight_decay=args.weight_decay,
        lr_anneal_steps=args.lr_anneal_steps,
    ).run_loop()


def create_argparser():
    defaults = dict(
        data_dir="/root/autodl-tmp/colon_seg_normal/colon_seg_normal_train/train",
        schedule_sampler="uniform",
        lr=1e-4,
        weight_decay=0.0,
        lr_anneal_steps=0,
        batch_size=16,
        microbatch=-1,  # -1 disables microbatches
        ema_rate="0.9999",  # comma-separated list of EMA values
        log_interval=10,
        save_interval=10000,
        resume_checkpoint="",
        use_fp16=False,
        fp16_scale_growth=1e-3,
    )
    defaults.update(model_and_diffusion_defaults())
    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()
