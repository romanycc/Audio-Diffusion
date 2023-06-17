# based on https://github.com/huggingface/diffusers/blob/main/examples/train_unconditional.py
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "6"
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





logger = get_logger(__name__)


def get_full_repo_name(model_id: str,
                       organization: Optional[str] = None,
                       token: Optional[str] = None):
    if token is None:
        token = HfFolder.get_token()
    if organization is None:
        username = whoami(token)["name"]
        return f"{username}/{model_id}"
    else:
        return f"{organization}/{model_id}"


def setup_logging(run_name):
    os.makedirs(os.path.join("models", run_name), exist_ok=True)
    os.makedirs(os.path.join("results", run_name), exist_ok=True)
    os.makedirs(os.path.join("results_image", run_name), exist_ok=True)

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


def main(args):
    output_dir = os.environ.get("SM_MODEL_DIR", None) or args.output_dir
    # logging_dir = os.path.join(output_dir, args.logging_dir)
    setup_logging(output_dir)
    logging_dir = f"logs/{output_dir}"

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with="tensorboard",
        logging_dir=logging_dir,
    )

# -------------------- load dataset ------------------------------ #

    if args.dataset_name is not None:
        if os.path.exists(args.dataset_name):
            dataset = load_from_disk(
                args.dataset_name,
                storage_options=args.dataset_config_name)["train"]
        else:
            dataset = load_dataset(
                args.dataset_name,
                args.dataset_config_name,
                cache_dir=args.cache_dir,
                use_auth_token=True if args.use_auth_token else None,
                split="train",
            )
    else:
        dataset = load_dataset(
            "imagefolder",
            data_dir=args.train_data_dir,
            cache_dir=args.cache_dir,
            split="train",
        )
    # Determine image resolution
    print(len(dataset))
    resolution = dataset[0]["image"].height, dataset[0]["image"].width

    code_dict = getTrainData()

# -------------------- 處理 dataset ------------------------------ #

    augmentations = Compose([
        ToTensor(),
        Normalize([0.5], [0.5]),
    ])
    

    def transforms(examples):
        if args.vae is not None and vqvae.config["in_channels"] == 3:
            images = [
                augmentations(image.convert("RGB"))
                for image in examples["image"]
            ]
        else:
            images = [augmentations(image) for image in examples["image"]]
        if args.encodings is not None:
            encoding = [encodings[file] for file in examples["audio_file"]]

            return {"input": images, "encoding": encoding}

        label = [file.split("-")[-1] for file in examples["audio_file"]]
        labelindex = [int(file[:-4]) for file in label]
        lb = preprocessing.LabelBinarizer()
        lb.fit([i for i in range(50)])
        one_hot = [lb.transform([index]) for index in labelindex]
        return {"input": images, "label":one_hot}

    dataset.set_transform(transforms)
    train_dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=args.train_batch_size, shuffle=True)

    if args.encodings is not None:
        encodings = pickle.load(open(args.encodings, "rb"))

    vqvae = None
    if args.vae is not None:
        try:
            vqvae = AutoencoderKL.from_pretrained(args.vae)
        except EnvironmentError:
            vqvae = AudioDiffusionPipeline.from_pretrained(args.vae).vqvae
        # Determine latent resolution
        with torch.no_grad():
            latent_resolution = vqvae.encode(
                torch.zeros((1, 1) +
                            resolution)).latent_dist.sample().shape[2:]

# -------------------- load model ------------------------------ #

    if args.from_pretrained is not None:
        pipeline = AudioDiffusionPipeline.from_pretrained(args.from_pretrained)
        mel = pipeline.mel
        model = pipeline.unet
        if hasattr(pipeline, "vqvae"):
            vqvae = pipeline.vqvae

    else:
        if args.encodings is None:
            model = UNet2DModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 128, 256, 256, 512, 512),
                down_block_types=(
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "DownBlock2D",
                    "AttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "AttnUpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                    "UpBlock2D",
                ),
                num_class_embeds = 50,
            )
            model.class_embedding = nn.Linear(50 ,512)
        else:
            model = UNet2DConditionModel(
                sample_size=resolution if vqvae is None else latent_resolution,
                in_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                out_channels=1
                if vqvae is None else vqvae.config["latent_channels"],
                layers_per_block=2,
                block_out_channels=(128, 256, 512, 512),
                down_block_types=(
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "CrossAttnDownBlock2D",
                    "DownBlock2D",
                ),
                up_block_types=(
                    "UpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                    "CrossAttnUpBlock2D",
                ),
                cross_attention_dim=list(encodings.values())[0].shape[-1],
            )

    if args.scheduler == "ddpm":
        noise_scheduler = DDPMScheduler(
            num_train_timesteps=args.num_train_steps)
    else:
        noise_scheduler = DDIMScheduler(
            num_train_timesteps=args.num_train_steps)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps,
        num_training_steps=(len(train_dataloader) * args.num_epochs) //
        args.gradient_accumulation_steps,
    )

    model, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        model, optimizer, train_dataloader, lr_scheduler)

    ema_model = EMAModel(
        getattr(model, "module", model),
        inv_gamma=args.ema_inv_gamma,
        power=args.ema_power,
        max_value=args.ema_max_decay,
    )

    if args.push_to_hub:
        if args.hub_model_id is None:
            repo_name = get_full_repo_name(Path(output_dir).name,
                                           token=args.hub_token)
        else:
            repo_name = args.hub_model_id
        repo = Repository(output_dir, clone_from=repo_name)

    if accelerator.is_main_process:
        run = os.path.split(__file__)[-1].split(".")[0]
        accelerator.init_trackers(run)

    mel = Mel(
        x_res=resolution[1],
        y_res=resolution[0],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )

# -------------------- training ------------------------------ #

    global_step = 0
    for epoch in range(args.num_epochs):
        progress_bar = tqdm(total=len(train_dataloader),
                            disable=not accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")

        if epoch < args.start_epoch:
            for step in range(len(train_dataloader)):
                optimizer.step()
                lr_scheduler.step()
                progress_bar.update(1)
                global_step += 1
            if epoch == args.start_epoch - 1 and args.use_ema:
                ema_model.optimization_step = global_step
            continue

        model.train()
        for step, batch in enumerate(train_dataloader):
            clean_images = batch["input"]
            one_hot = batch["label"]


            if vqvae is not None:
                vqvae.to(clean_images.device)
                with torch.no_grad():
                    clean_images = vqvae.encode(
                        clean_images).latent_dist.sample()
                # Scale latent images to ensure approximately unit variance
                clean_images = clean_images * 0.18215

            # Sample noise that we'll add to the images
            noise = torch.randn(clean_images.shape).to(clean_images.device)
            bsz = clean_images.shape[0]
            # Sample a random timestep for each image
            timesteps = torch.randint(
                0,
                noise_scheduler.config.num_train_timesteps,
                (bsz, ),
                device=clean_images.device,
            ).long()

            # Add noise to the clean images according to the noise magnitude at each timestep
            # (this is the forward diffusion process)
            noisy_images = noise_scheduler.add_noise(clean_images, noise,
                                                     timesteps)

            with accelerator.accumulate(model):
                # Predict the noise residual
                if args.encodings is not None:
                    noise_pred = model(noisy_images, timesteps,
                                       batch["encoding"])["sample"]
                else:
                    #print(noisy_images.shape, one_hot.shape)
                    one_hot = one_hot.squeeze()
                    #print(one_hot.shape)
                    noise_pred = model(noisy_images, timesteps, class_labels = one_hot.to(torch.float32))["sample"]
                
                if args.loss == "l2":
                    loss = F.mse_loss(noise_pred, noise)
                elif args.loss == "l1":
                    loss = F.l1_loss(noise_pred, noise)

                accelerator.backward(loss)

                if accelerator.sync_gradients:
                    accelerator.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                lr_scheduler.step()
                if args.use_ema:
                    ema_model.step(model)
                optimizer.zero_grad()

            progress_bar.update(1)
            global_step += 1

            logs = {
                "loss": loss.detach().item(),
                "lr": lr_scheduler.get_last_lr()[0],
                "step": global_step,
            }
            if args.use_ema:
                logs["ema_decay"] = ema_model.decay
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)
        progress_bar.close()

        accelerator.wait_for_everyone()

        # -------------------- 生成聲音、影像 儲存model------------------------------ #
        
        if accelerator.is_main_process:
            if ((epoch + 1) % args.save_model_epochs == 0
                    or (epoch + 1) % args.save_images_epochs == 0
                    or epoch == args.num_epochs - 1):
                unet = accelerator.unwrap_model(model)
                if args.use_ema:
                    ema_model.copy_to(unet.parameters())
                pipeline = AudioDiffusionPipeline(
                    vqvae=vqvae,
                    unet=unet,
                    mel=mel,
                    scheduler=noise_scheduler,
                )

            if (epoch + 1) % args.save_model_epochs == 0 or epoch == args.num_epochs - 1:
                pipeline.save_pretrained("models/"+output_dir)

                # save the model
                if args.push_to_hub:
                    repo.push_to_hub(
                        commit_message=f"Epoch {epoch}",
                        blocking=False,
                        auto_lfs_prune=True,
                    )

            if (epoch + 1) % args.save_images_epochs == 0:
                generator = torch.Generator(
                    device=clean_images.device).manual_seed(42)

                if args.encodings is not None:
                    random.seed(42)
                    encoding = torch.stack(
                        random.sample(list(encodings.values()),
                                      args.eval_batch_size)).to(
                                          clean_images.device)
                else:
                    encoding = None


                # 選擇eval batch size個label，並做onehot
                lb = preprocessing.LabelBinarizer()
                lb.fit([i for i in range(50)])
                one_hot = [lb.transform([i]) for i in range(args.eval_batch_size)]
                label = [code_dict[i] for i in range(args.eval_batch_size)] # 讀取類別
                one_hot = torch.tensor(one_hot).squeeze()

                # run pipeline in inference (sample random noise and denoise)
                images, (sample_rate, audios) = pipeline(
                    generator=generator,
                    batch_size=args.eval_batch_size,
                    return_dict=False,
                    encoding=encoding,
                    one_hot = one_hot,
                    #class_labels = torch.tensor(one_hot).to(torch.float32)
                )

                # denormalize the images and save to tensorboard
                # images = np.array([
                #     np.frombuffer(image.tobytes(), dtype="uint8").reshape(
                #         (len(image.getbands()), image.height, image.width))
                #     for image in images
                # ])
                index = 0
                for image in images:
                    im = Image.fromarray(np.uint8(image))
                    im.save(f"results_image/{output_dir}/epoch{epoch}_{label[index]}.png")
                    index += 1

                # accelerator.trackers[0].writer.add_images(
                #     "test_samples", images, epoch)
                index = 0
                for _, audio in enumerate(audios):
                    sf.write( f"results/{output_dir}/epoch{epoch}_{label[index]}.wav", audio, sample_rate)
                    index += 1
                    # accelerator.trackers[0].writer.add_audio(
                    #     f"test_audio_{_}",
                    #     normalize(audio),
                    #     epoch,
                    #     sample_rate=sample_rate,
                    # )
        accelerator.wait_for_everyone()

    accelerator.end_training()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Simple example of a training script.")
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--dataset_name", type=str, default=None)
    parser.add_argument("--dataset_config_name", type=str, default=None)
    parser.add_argument(
        "--train_data_dir",
        type=str,
        default="ESC-50-master/audio_spec/train",
        help="A folder containing the training data.",
    )
    parser.add_argument("--output_dir", type=str, default="ddpm-model-64")
    parser.add_argument("--overwrite_output_dir", type=bool, default=False)
    parser.add_argument("--cache_dir", type=str, default=None)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--eval_batch_size", type=int, default=16)
    parser.add_argument("--num_epochs", type=int, default=100)
    parser.add_argument("--save_images_epochs", type=int, default=10) #10
    parser.add_argument("--save_model_epochs", type=int, default=10)  #10
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-4)
    parser.add_argument("--lr_scheduler", type=str, default="cosine")
    parser.add_argument("--lr_warmup_steps", type=int, default=500)
    parser.add_argument("--adam_beta1", type=float, default=0.95)
    parser.add_argument("--adam_beta2", type=float, default=0.999)
    parser.add_argument("--adam_weight_decay", type=float, default=1e-6)
    parser.add_argument("--adam_epsilon", type=float, default=1e-08)
    parser.add_argument("--use_ema", type=bool, default=True)
    parser.add_argument("--ema_inv_gamma", type=float, default=1.0)
    parser.add_argument("--ema_power", type=float, default=3 / 4)
    parser.add_argument("--ema_max_decay", type=float, default=0.9999)
    parser.add_argument("--push_to_hub", type=bool, default=False)
    parser.add_argument("--use_auth_token", type=bool, default=False)
    parser.add_argument("--hub_token", type=str, default=None)
    parser.add_argument("--hub_model_id", type=str, default=None)
    parser.add_argument("--hub_private_repo", type=bool, default=False)
    parser.add_argument("--logging_dir", type=str, default="logs")
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default="no",
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose"
            "between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >= 1.10."
            "and an Nvidia Ampere GPU."),
    )
    parser.add_argument("--hop_length", type=int, default=1024)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    parser.add_argument("--from_pretrained", type=str, default=None)
    parser.add_argument("--start_epoch", type=int, default=0)
    parser.add_argument("--num_train_steps", type=int, default=1000)
    parser.add_argument("--scheduler",
                        type=str,
                        default="ddpm",
                        help="ddpm or ddim")
    parser.add_argument(
        "--vae",
        type=str,
        default=None,
        help="pretrained VAE model for latent diffusion",
    )
    parser.add_argument(
        "--encodings",
        type=str,
        default=None,
        help="picked dictionary mapping audio_file to encoding",
    )
    parser.add_argument(
        "--loss",
        type=str,
        default="l2",
    )

    args = parser.parse_args()
    env_local_rank = int(os.environ.get("LOCAL_RANK", -1))
    if env_local_rank != -1 and env_local_rank != args.local_rank:
        args.local_rank = env_local_rank

    if args.dataset_name is None and args.train_data_dir is None:
        raise ValueError(
            "You must specify either a dataset name from the hub or a train data directory."
        )

    main(args)
