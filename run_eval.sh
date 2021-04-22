pretrained_path=BEST_MODEL
source_path=./dataset/cmu_us_ksp_arctic/wav/arctic_a0003.wav
target_path=./dataset/cmu_us_bdl_arctic/wav/arctic_a0003.wav

python eval.py -m $pretrained_path -s $source_path -t $target_path