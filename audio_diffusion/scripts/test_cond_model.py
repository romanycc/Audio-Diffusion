# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import numpy as np
import json
import soundfile as sf
import argparse
from sklearn import preprocessing
import librosa
import torch
cpu_num = 6  # Num of CPUs you want to use
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import csv
from torchvision import datasets, transforms
from torchvision.utils import save_image
from torch.optim.lr_scheduler import StepLR
from diffusers import UNet2DConditionModel, UNet2DModel
from torch.utils.data import SubsetRandomSampler
import random
import math
from accelerate import Accelerator
#from dataloader_b import encoding_dict
from diffusers.optimization import get_cosine_schedule_with_warmup
from diffusers import DiffusionPipeline
import torchvision
import time
from tqdm import tqdm
from audio_diffusion_pytorch import DiffusionVocoder,DiffusionModel, UNetV0, VDiffusion, VSampler
from torch.utils.tensorboard import SummaryWriter
from audio_diffusion_pytorch import DiffusionAE, UNetV0, VDiffusion, VSampler
from audio_encoders_pytorch import MelE1d, TanhBottleneck


import argparse
import os
import pickle
import random
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from accelerate import Accelerator
from accelerate.logging import get_logger
from datasets import load_dataset, load_from_disk
from diffusers import (AutoencoderKL, DDIMScheduler, DDPMScheduler,
                       UNet2DConditionModel, UNet2DModel)
from diffusers.optimization import get_scheduler
from diffusers.pipelines.audio_diffusion import Mel
from diffusers.training_utils import EMAModel
from huggingface_hub import HfFolder, Repository, whoami
from librosa.util import normalize
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm.auto import tqdm
import fnmatch
import importlib
import inspect
import os
import re
import sys
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import PIL
import torch
from huggingface_hub import hf_hub_download, model_info, snapshot_download
from packaging import version
from tqdm.auto import tqdm

import diffusers
from diffusers.configuration_utils import ConfigMixin
from diffusers.models.modeling_utils import _LOW_CPU_MEM_USAGE_DEFAULT
from diffusers.schedulers.scheduling_utils import SCHEDULER_CONFIG_NAME
from diffusers.utils import (
    CONFIG_NAME,
    DEPRECATED_REVISION_ARGS,
    DIFFUSERS_CACHE,
    HF_HUB_OFFLINE,
    SAFETENSORS_WEIGHTS_NAME,
    WEIGHTS_NAME,
    BaseOutput,
    deprecate,
    get_class_from_dynamic_module,
    is_accelerate_available,
    is_accelerate_version,
    is_compiled_module,
    is_safetensors_available,
    is_torch_version,
    is_transformers_available,
    logging,
    numpy_to_pil,
)
'''overwrite AudioDiffusionPipeline start'''
#from audiodiffusion.pipeline_audio_diffusion import AudioDiffusionPipeline 
# This code has been migrated to diffusers but can be run locally with
# pipe = DiffusionPipeline.from_pretrained("teticio/audio-diffusion-256", custom_pipeline="audio-diffusion/audiodiffusion/pipeline_audio_diffusion.py")

# Copyright 2022 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from sklearn import preprocessing
from math import acos, sin
from typing import List, Tuple, Union

import numpy as np
import torch
from diffusers import (
    AudioPipelineOutput,
    AutoencoderKL,
    DDIMScheduler,
    DDPMScheduler,
    DiffusionPipeline,
    ImagePipelineOutput,
    UNet2DConditionModel,
)
from diffusers.utils import BaseOutput
from PIL import Image

from mel import Mel

class AudioDiffusionPipeline(DiffusionPipeline):
    """
    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Parameters:
        vqae ([`AutoencoderKL`]): Variational AutoEncoder for Latent Audio Diffusion or None
        unet ([`UNet2DConditionModel`]): UNET model
        mel ([`Mel`]): transform audio <-> spectrogram
        scheduler ([`DDIMScheduler` or `DDPMScheduler`]): de-noising scheduler
    """

    _optional_components = ["vqvae"]

    def __init__(
        self,
        vqvae: AutoencoderKL,
        unet: UNet2DConditionModel,
        mel: Mel,
        scheduler: Union[DDIMScheduler, DDPMScheduler],
    ):


        super().__init__()
        self.register_modules(unet=unet, scheduler=scheduler, mel=mel, vqvae=vqvae)
        unet.class_embedding = nn.Linear(50 ,512)
        state_dict = torch.load("models/roman_cond_L2/unet/diffusion_pytorch_model.bin")
        filtered_state_dict = {k[16:]: v for k, v in state_dict.items() if k =="class_embedding.weight" or k=="class_embedding.bias"}
        unet.class_embedding.load_state_dict(filtered_state_dict)
        print(unet.class_embedding)


    def get_default_steps(self) -> int:
        """Returns default number of steps recommended for inference

        Returns:
            `int`: number of steps
        """
        return 50 if isinstance(self.scheduler, DDIMScheduler) else 1000

    @torch.no_grad()
    def __call__(
        self,
        batch_size: int = 1,
        audio_file: str = None,
        raw_audio: np.ndarray = None,
        slice: int = 0,
        start_step: int = 0,
        steps: int = None,
        generator: torch.Generator = None,
        mask_start_secs: float = 0,
        mask_end_secs: float = 0,
        step_generator: torch.Generator = None,
        eta: float = 0,
        noise: torch.Tensor = None,
        encoding: torch.Tensor = None,
        return_dict=True,
        one_hot = None,
    ) -> Union[
        Union[AudioPipelineOutput, ImagePipelineOutput],
        Tuple[List[Image.Image], Tuple[int, List[np.ndarray]]],
    ]:
        """Generate random mel spectrogram from audio input and convert to audio.

        Args:
            batch_size (`int`): number of samples to generate
            audio_file (`str`): must be a file on disk due to Librosa limitation or
            raw_audio (`np.ndarray`): audio as numpy array
            slice (`int`): slice number of audio to convert
            start_step (int): step to start from
            steps (`int`): number of de-noising steps (defaults to 50 for DDIM, 1000 for DDPM)
            generator (`torch.Generator`): random number generator or None
            mask_start_secs (`float`): number of seconds of audio to mask (not generate) at start
            mask_end_secs (`float`): number of seconds of audio to mask (not generate) at end
            step_generator (`torch.Generator`): random number generator used to de-noise or None
            eta (`float`): parameter between 0 and 1 used with DDIM scheduler
            noise (`torch.Tensor`): noise tensor of shape (batch_size, 1, height, width) or None
            encoding (`torch.Tensor`): for UNet2DConditionModel shape (batch_size, seq_length, cross_attention_dim)
            return_dict (`bool`): if True return AudioPipelineOutput, ImagePipelineOutput else Tuple

        Returns:
            `List[PIL Image]`: mel spectrograms (`float`, `List[np.ndarray]`): sample rate and raw audios
        """

        steps = steps or self.get_default_steps()
        self.scheduler.set_timesteps(steps)
        step_generator = step_generator or generator
        # For backwards compatibility
        if type(self.unet.sample_size) == int:
            self.unet.sample_size = (self.unet.sample_size, self.unet.sample_size)
        if noise is None:
            noise = torch.randn(
                (
                    batch_size,
                    self.unet.in_channels,
                    self.unet.sample_size[0],
                    self.unet.sample_size[1],
                ),
                generator=generator,
                device=self.device,
            )
        images = noise
        mask = None

        if audio_file is not None or raw_audio is not None:
            self.mel.load_audio(audio_file, raw_audio)
            input_image = self.mel.audio_slice_to_image(slice)
            input_image = np.frombuffer(input_image.tobytes(), dtype="uint8").reshape(
                (input_image.height, input_image.width)
            )
            input_image = (input_image / 255) * 2 - 1
            input_images = torch.tensor(input_image[np.newaxis, :, :], dtype=torch.float).to(self.device)

            if self.vqvae is not None:
                input_images = self.vqvae.encode(torch.unsqueeze(input_images, 0)).latent_dist.sample(
                    generator=generator
                )[0]
                input_images = 0.18215 * input_images

            if start_step > 0:
                images[0, 0] = self.scheduler.add_noise(input_images, noise, self.scheduler.timesteps[start_step - 1])

            pixels_per_second = (
                self.unet.sample_size[1] * self.mel.get_sample_rate() / self.mel.x_res / self.mel.hop_length
            )
            mask_start = int(mask_start_secs * pixels_per_second)
            mask_end = int(mask_end_secs * pixels_per_second)
            mask = self.scheduler.add_noise(input_images, noise, torch.tensor(self.scheduler.timesteps[start_step:]))

        for step, t in enumerate(self.progress_bar(self.scheduler.timesteps[start_step:])):
            if isinstance(self.unet, UNet2DConditionModel):
                model_output = self.unet(images, t, encoding)["sample"]
            else:
                # lb = preprocessing.LabelBinarizer()
                # lb.fit([i for i in range(50)])
                # one_hot = [lb.transform([i]) for i in range(args.eval_batch_size)]
                # one_hot = torch.tensor(one_hot).squeeze()
                model_output = self.unet(images, t, one_hot.to(torch.float32).to(self.device))["sample"]

            if isinstance(self.scheduler, DDIMScheduler):
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    eta=eta,
                    generator=step_generator,
                )["prev_sample"]
            else:
                images = self.scheduler.step(
                    model_output=model_output,
                    timestep=t,
                    sample=images,
                    generator=step_generator,
                )["prev_sample"]

            if mask is not None:
                if mask_start > 0:
                    images[:, :, :, :mask_start] = mask[:, step, :, :mask_start]
                if mask_end > 0:
                    images[:, :, :, -mask_end:] = mask[:, step, :, -mask_end:]

        if self.vqvae is not None:
            # 0.18215 was scaling factor used in training to ensure unit variance
            images = 1 / 0.18215 * images
            images = self.vqvae.decode(images)["sample"]

        images = (images / 2 + 0.5).clamp(0, 1)
        images = images.cpu().permute(0, 2, 3, 1).numpy()
        images = (images * 255).round().astype("uint8")
        images = list(
            map(lambda _: Image.fromarray(_[:, :, 0]), images)
            if images.shape[3] == 1
            else map(lambda _: Image.fromarray(_, mode="RGB").convert("L"), images)
        )

        audios = list(map(lambda _: self.mel.image_to_audio(_), images))
        if not return_dict:
            return images, (self.mel.get_sample_rate(), audios)

        return BaseOutput(**AudioPipelineOutput(np.array(audios)[:, np.newaxis, :]), **ImagePipelineOutput(images))

    @torch.no_grad()
    def encode(self, images: List[Image.Image], steps: int = 50) -> np.ndarray:
        """Reverse step process: recover noisy image from generated image.

        Args:
            images (`List[PIL Image]`): list of images to encode
            steps (`int`): number of encoding steps to perform (defaults to 50)

        Returns:
            `np.ndarray`: noise tensor of shape (batch_size, 1, height, width)
        """

        # Only works with DDIM as this method is deterministic
        assert isinstance(self.scheduler, DDIMScheduler)
        self.scheduler.set_timesteps(steps)
        sample = np.array(
            [np.frombuffer(image.tobytes(), dtype="uint8").reshape((1, image.height, image.width)) for image in images]
        )
        sample = (sample / 255) * 2 - 1
        sample = torch.Tensor(sample).to(self.device)

        for t in self.progress_bar(torch.flip(self.scheduler.timesteps, (0,))):
            prev_timestep = t - self.scheduler.config.num_train_timesteps // self.scheduler.num_inference_steps
            alpha_prod_t = self.scheduler.alphas_cumprod[t]
            alpha_prod_t_prev = (
                self.scheduler.alphas_cumprod[prev_timestep]
                if prev_timestep >= 0
                else self.scheduler.final_alpha_cumprod
            )
            beta_prod_t = 1 - alpha_prod_t
            model_output = self.unet(sample, t)["sample"]
            pred_sample_direction = (1 - alpha_prod_t_prev) ** (0.5) * model_output
            sample = (sample - pred_sample_direction) * alpha_prod_t_prev ** (-0.5)
            sample = sample * alpha_prod_t ** (0.5) + beta_prod_t ** (0.5) * model_output

        return sample

    @staticmethod
    def slerp(x0: torch.Tensor, x1: torch.Tensor, alpha: float) -> torch.Tensor:
        """Spherical Linear intERPolation

        Args:
            x0 (`torch.Tensor`): first tensor to interpolate between
            x1 (`torch.Tensor`): seconds tensor to interpolate between
            alpha (`float`): interpolation between 0 and 1

        Returns:
            `torch.Tensor`: interpolated tensor
        """

        theta = acos(torch.dot(torch.flatten(x0), torch.flatten(x1)) / torch.norm(x0) / torch.norm(x1))
        return sin((1 - alpha) * theta) * x0 / sin(theta) + sin(alpha * theta) * x1 / sin(theta)
    

'''overwrite AudioDiffusionPipeline Done'''

def getTrainData():
    fileroot = "ESC-50-master/meta/esc50.csv"
    sound_filename = []
    sound_label = []
    skip_first_row = True
    code_dict = {}

    with open(fileroot, 'r') as file:
        csv_reader = csv.reader(file)
        for row in csv_reader:      # row is a list : filename	fold	target	category	esc10	src_file	take
            if skip_first_row:
                skip_first_row = False
            else:
                code_dict[int(row[2])] = row[3]

    return code_dict


'''
使用說明
1. 更改程式內 parser 參數
2. 更改程式內 line 171 的 unet 權重檔路徑
3. 更改 pretrain路徑內 的 model_index.json檔 的 mel 為

    "mel": [
        "audio_diffusion",  # 原本是 null 改成 "audio_diffusion"
        "Mel"
    ],

'''

parser = argparse.ArgumentParser(description="Simple example of a training script.")
parser.add_argument("--output_dir", type=str, default="2500mel")
parser.add_argument("--eval_batch_size", type=int, default=50)
parser.add_argument("--from_pretrained", type=str, default="models/roman_cond_L2") #line 171 also need to modify
parser.add_argument("--device", type=str, default="cuda")
args = parser.parse_args()
pipeline = AudioDiffusionPipeline.from_pretrained(pretrained_model_name_or_path = args.from_pretrained, low_cpu_mem_usage=False)
print(pipeline)
pipeline = pipeline.to(args.device)
generator = torch.Generator(
    device=args.device).manual_seed(42)###
encoding = None
# 選擇eval batch size個label，並做onehot
lb = preprocessing.LabelBinarizer()
lb.fit([i for i in range(50)])
one_hot = [lb.transform([i]) for i in range(args.eval_batch_size)]
code_dict = getTrainData()
label = [code_dict[i] for i in range(args.eval_batch_size)] # 讀取類別
one_hot = torch.tensor(one_hot).squeeze()

# run pipeline in inference (sample random noise and denoise)
images, (sample_rate, audios) = pipeline(
    generator=generator,
    batch_size=args.eval_batch_size,
    return_dict=False,
    encoding=encoding,
    one_hot = one_hot,
)

index = 0
for image in images:
    im = Image.fromarray(np.uint8(image))
    im.save(f"results_image/{args.output_dir}/test_{label[index]}.png")
    index += 1

index = 0
for _, audio in enumerate(audios):
    sf.write( f"results/{args.output_dir}/test_{label[index]}.wav", audio, sample_rate)
    index += 1
