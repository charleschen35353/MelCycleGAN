# MelCycleGAN

## Description

In this project, we adopt the method that is used on MelGan-VC to convert audio from Indian accents to American accents.  

We adopt Mel freqeuncy spectrum as our training feature, and apply a cycle GAN with sharing siamese network to perform domain transformation. 

Our project demonstrate a more stable and quick way to perform domain trainsformation in between two domains of audio data.

For more details, please refer to my paper here:

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

## Pretrain model

Please download and unzip this fhttps://drive.google.com/drive/folders/13TC3CPbYN44K_5dkq0F07pN7ujouMDU8?usp=sharing

## Train

```
python train.py [-s source wav dataset path] [-t target wav dataset path] 
```

for example, 

```
python train.py -s ./dataset/cmu_us_ksp_arctic/wav/ -t ./dataset/cmu_us_bdl_arctic/wav/
```

## Eval

```
python eval.py [-m model folder] [-s source wav path] [-t target wav path]
```

for example, 
```
python eval.py -m MELGANVC-best -s ./dataset/cmu_us_ksp_arctic/wav/arctic_a0003.wav -t ./dataset/cmu_us_bdl_arctic/wav/arctic_a0003.wav
```

