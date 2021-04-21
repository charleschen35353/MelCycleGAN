import librosa
import soundfile as sf
import os
from spec_utils import *
from utils import *
from models import *


#Converting from source Spectrogram to target Spectrogram
def convert(spec_source, spec_target, generator, output_name, output_path='./', sr = 16000, show=False):
  specarr_s = chopspec(spec_source)
  specarr_t = chopspec(spec_target)
  print('Generating...')
  generated = generator(specarr_s, training=False)
  print('Assembling and Converting...')
  specarr_s = specass(specarr_s, spec_source)
  specarr_t = specass(specarr_t, spec_source)
  generated = specass(generated, spec_source)
  specarr_s_wv = deprep(specarr_s)
  specarr_t_wv = deprep(specarr_t)
  generated_wv = deprep(generated)
  print('Saving...')
  pathfin = f'{output_path}/{output_name}'
  if not os.path.exists(pathfin):
    os.mkdir(pathfin)
  sf.write(pathfin+'/Generated.wav', generated_wv, sr)
  sf.write(pathfin+'/Original.wav', specarr_s_wv, sr)
  print('Saved WAV!')


weights = "MELGANVC-best"
i = 2


#Wav to wav conversion
audio_pth_a = (os.path.join(root_dir, "dataset/cmu_us_ksp_arctic/wav/arctic_a000{}.wav").format(i))
audio_pth_b = (os.path.join(root_dir, "dataset/cmu_us_bdl_arctic/wav/arctic_a000{}.wav").format(i))

model_info = get_networks(shape, load_model = True, path = os.path.join(root_dir,weights))
bdl2ksp_gen, _ , ksp2bdl_gen, _, _, _ = model_info

wv_a, sr_a = librosa.load(audio_pth_a, sr=sr)  #Load waveform
speca = prep(wv_a)
    
wv_b, sr_a = librosa.load(audio_pth_b, sr=sr)  #Load waveform
specb = prep(wv_b)                                               #Waveform to Spectrogram

converted = convert(speca, specb, ksp2bdl_gen, "result{}".format(i), output_path=root_dir, sr = sr, show=False)
converted = convert(specb, speca, bdl2ksp_gen, "result{}".format(i), output_path=root_dir, sr = sr, show=False)

