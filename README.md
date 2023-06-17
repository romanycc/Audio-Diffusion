# Audio Diffusion Research

* Team members : 楊佳誠、邱以中、蔡桔析

## Introduction

* This is the final project for NYCU_DLP course.
* After reading the paper"Palette:ASimple,General Framework for Image-to-Image Translation," we found it interesting to investigate whether the ablation results are consistent with those in the audio domain.
* We will compare the L1 and L2 loss and also evaluatethe significance of self-attention and normalization in the audio diffusion architecture.

## Method
* Using this github to implement our project：https://github.com/teticio/audio-diffusion
#### Model 
* The audio diffusion backbone utilizes a U-Net architecture.
* The class_embedding of the UNet2DModel is utilized to incorporate both the time embedding and class_embedding, treating them as part of the conditional input of the model.
#### Dataset
* ESC-50 consists of 5-second-long recordings organized into 50 semantical classes, with 40 examples per class.
* This dataset consists of the following five main categories: Animals, Natural, Human non-speech sounds, Interior sounds, and Exterior noises.
#### Data preprocessing
* The .wav data is preprocessed into Mel Spectrograms. This can be done by audio_diffusion/scripts/audio_to_images.py. 
* The Mel spectrograms will be normalized according to the experiment setting.

## Training
* We write our code in the audio_diffusion/scripts folder.
* Use train_unet.py to train our model
* Load the preprocess data folder by the path where audio_to_images.py generates.
## Sampling
* Use audio_diffusion/scripts/test_cond_model.py to generate sample. This program generate 40 .wav files for 50 classes in ESC-50.
* There are several things need to modified before you run this code：
    1. Modify the parameter of parser
    2. Replace the path in line 171 by your pretrained unet weight.(ex. /unet/diffusion_pytorch_model.bin)
    3. Modify the model_index.json file in your saved model path：
```json
    "mel": [
        "audio_diffusion", # change null to "audio_diffusion"
        "Mel"
    ],
```
## Evaluation

* We utilize our model to generate 50 classes of audio, producing 40 audio samples for each class as evaluation data.
* The FAD score is a metric employed to measure the similarity between evaluation data and original data. A lower FAD score indicates a closer match between the distributions of the generated and real audio.
* The CA score, utilizing pretrained Contrastive Language-Audio Pretraining (CLAP), is used to assess whether our model can successfully generate the correct voice.

#### Expected file
* To compute FAD and CA, the path should contain 50 folder, which is named from 0 to 49 by its label. Each folder should contain 40 .wav generate from the same class.
* Take a look at audio_evaluate/Predict/L2 as an example.

#### FAD
* Use audio_evaluate/evaluate.py to compute FAD score.
```
dir_1 = path of ground truth .wav.
dir_2 = path of generate .wav.
```
#### CA
* Download this github：https://github.com/LAION-AI/CLAP
* Use audio_evaluate/result/CA/zero-shot-classification/CLASS.PY to compute CA.
* Go to https://huggingface.co/lukewys/laion_clap/blob/main/630k-audioset-best.pt download the pretrained weight.
* Put your generate file path to esc50_test_dir

## Demo


https://github.com/romanycc/Audio-Diffusion/assets/131567914/a0e34696-531f-4b51-a450-af0c737d993e


https://github.com/romanycc/Audio-Diffusion/assets/131567914/d8a052d8-9e55-4165-83f9-3f3373926226


https://github.com/romanycc/Audio-Diffusion/assets/131567914/6b08b2e3-c69a-4ad5-b779-1567948cab16







## Reference

1. https://github.com/teticio/audio-diffusion
2. https://github.com/LAION-AI/CLAP
3. https://github.com/gudgud96/frechet-audio-distance
4. https://github.com/huggingface/diffusers
5. https://arxiv.org/abs/2111.05826


