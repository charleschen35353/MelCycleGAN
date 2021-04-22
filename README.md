# MelCycleGAN

## Description

In this project, we adopt the method that is used on MelGan-VC to convert audio from Indian accents to American accents.  

MelCycleGAN is based on MelGAN-VC which is proposed by Marco Pasini in 2019. 

Our method speed up the training time and make the entire training more stable by adopting a CycleGAN structure with shared Siamese training.

For more details, please refer to my paper here: [LINK]

## Presentation and Demo:

https://youtu.be/Q1v5FnGqSGs

## Results:

Results can be found in ./results.

For example,

```indian-1.wav``` is the audio in source indian accent.
```american-1.wav``` is the audio in target american accent.
```gan-1.wav``` is the converted audio using MelGAN-VC.
```cycle-1.wav``` is the converted audio using our method MelCycleGAN.

And for quailty evaluation,

```gan_epoch1.wav```is the audio generated from random sample using MelGAN-VC with 1 training epoch.

```cycle_epoch1.wav```is the audio generated from random sample using our MelCycleGAN with 1 training epoch.

## Configuration

Our project depends on the following:

```
pip install tensorflow==2.4.0
pip install soundfile
pip intsall --no-deps torchaudio==0.5.0
pip install tensorflow-addons
pip install librosa
pip install torch==1.8.0+cpu torchvision==0.9.0+cpu torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
sudo apt-get install libsndfile1
```

If you have a GPU, please also make sure you have right CUDA version installed.

## Dataset

This project uses the CMU ARCTIC dataset, in specific the dataset with Indian English speaker and American English speaker. There are 1131+1 .wav audio files in each category. Please proceed to http://www.festvox.org/cmu_arctic/ and download and unzip them under the folder dataset.

## Pretrained model

Please download and unzip this https://drive.google.com/drive/folders/13TC3CPbYN44K_5dkq0F07pN7ujouMDU8?usp=sharing

## Instructions

We provide a script to run and evaluate our method. To reproduce our expriment, please run 

```
chmod +x run_all.sh
run_all.sh 
```

### Train

```
chmod +x run_train.sh
./run_train.sh
```

### Eval

```
chmod +x run_eval.sh
./run_eval.sh
```

You may also tune hyperparameters in the scripts above. 

# Notebook

We also provide a jupyter notebook to let you run on colab if the scripts or environment does not work.

# Citation

Our code is modified and refactorized from https://github.com/marcoppasini/MelGAN-VC.

