# Author: Tony Xu
#
# This code is adapted from the original DINOv2 repository: https://github.com/facebookresearch/dinov2
# This code is licensed under the CC BY-NC-ND 4.0 license
# found in the LICENSE file in the root directory of this source tree.

import argparse
import os
from typing import Any, List, Optional, Tuple

import torch
import torch.backends.cudnn as cudnn

from dinov2.models import build_model_from_cfg
from dinov2.utils.config import setup_3d
import dinov2.utils.utils as dinov2_utils


def get_args_parser(
    description: Optional[str] = None,
    parents: Optional[List[argparse.ArgumentParser]] = None,
    add_help: bool = True,
):
    parser = argparse.ArgumentParser(
        description=description,
        parents=parents or [],
        add_help=add_help,
    )
    parser.add_argument(
        "--config-file",
        type=str,
        help="Model configuration file",
    )
    parser.add_argument(
        "--pretrained-weights",
        type=str,
        help="Pretrained model weights",
    )
    parser.add_argument(
        "--strict-pretrained",
        action="store_true",
        help="Fail fast if pretrained checkpoint is missing or cannot be loaded.",
    )
    parser.add_argument(
        "--output-dir",
        default="",
        type=str,
        help="Output directory to write results and logs",
    )
    parser.add_argument(
        "--opts",
        help="Extra configuration options",
        default=[],
        nargs="+",
    )
    return parser


def get_autocast_dtype(config):
    teacher_dtype_str = config.compute_precision.teacher.backbone.mixed_precision.param_dtype
    if teacher_dtype_str == "fp16":
        return torch.half
    elif teacher_dtype_str == "bf16":
        return torch.bfloat16
    else:
        return torch.float


def build_model_for_eval(config, pretrained_weights, strict_pretrained=False):
    model, _ = build_model_from_cfg(config, only_teacher=True)
    try:
        if strict_pretrained and (not pretrained_weights or not os.path.isfile(pretrained_weights)):
            raise FileNotFoundError(f"Pretrained checkpoint not found: {pretrained_weights}")
        dinov2_utils.load_pretrained_weights(model, pretrained_weights, "teacher")
    except FileNotFoundError as e:
        if strict_pretrained:
            raise FileNotFoundError(
                f"Strict pretrained mode is enabled but checkpoint loading failed: {pretrained_weights}"
            ) from e
        print(e)
        print('No weights found, using random initialization!')
    model.eval()
    model.cuda()
    return model


def setup_and_build_model_3d(args) -> Tuple[Any, torch.dtype]:
    cudnn.benchmark = True
    config = setup_3d(args)
    model = build_model_for_eval(config, args.pretrained_weights, strict_pretrained=getattr(args, "strict_pretrained", False))
    autocast_dtype = get_autocast_dtype(config)
    return model, autocast_dtype
