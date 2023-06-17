import argparse
import io
import logging
import os
import re
import json
import numpy as np
import pandas as pd
from datasets import Dataset, DatasetDict, Features, Image, Value
from diffusers.pipelines.audio_diffusion import Mel
from tqdm.auto import tqdm

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger("audio_to_images")

with open("class_labels/ESC50_class_labels_indices_space.json", 'r') as file:
    code_dict = json.load(file)
print(code_dict)
code_dict = {value: key for key, value in code_dict.items()}


def main(args):
    mel = Mel(
        x_res=args.resolution[0],
        y_res=args.resolution[1],
        hop_length=args.hop_length,
        sample_rate=args.sample_rate,
        n_fft=args.n_fft,
    )
    os.makedirs(args.output_dir, exist_ok=True)
    audio_files = []
    print(args.input_dir)
    for root, _, files in os.walk(args.input_dir):
        for file in files:
            audio_files.append(root +"/" +file)
            #print((file.split("-")[-1])[:-4])
   
    for i in range(len(audio_files)):
        for j in range(len(audio_files)):
            if int((audio_files[j].split("-")[-1])[:-4]) == i:
                mel.load_audio(audio_files[j])
                for slice in range(mel.get_number_of_slices()):
                    image = mel.audio_slice_to_image(slice)
                    image.save("spec/" + code_dict[i] + ".png")
                break
            
  

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create dataset of Mel spectrograms from directory of audio files.")
    parser.add_argument("--input_dir", type=str, default="ESC-50-master/audio")
    parser.add_argument("--output_dir", type=str, default="ESC-50-master/audio_spec_save")
    parser.add_argument(
        "--resolution",
        type=str,
        default="64",
        help="Either square resolution or width,height.",
    )
    parser.add_argument("--hop_length", type=int, default=1024)
    parser.add_argument("--push_to_hub", type=str, default=None)
    parser.add_argument("--sample_rate", type=int, default=22050)
    parser.add_argument("--n_fft", type=int, default=2048)
    args = parser.parse_args()

    if args.input_dir is None:
        raise ValueError("You must specify an input directory for the audio files.")

    # Handle the resolutions.
    try:
        args.resolution = (int(args.resolution), int(args.resolution))
    except ValueError:
        try:
            args.resolution = tuple(int(x) for x in args.resolution.split(","))
            if len(args.resolution) != 2:
                raise ValueError
        except ValueError:
            raise ValueError("Resolution must be a tuple of two integers or a single integer.")
    assert isinstance(args.resolution, tuple)

    main(args)
