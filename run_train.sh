source_path=./dataset/cmu_us_ksp_arctic/wav
target_path=./dataset/cmu_us_bdl_arctic/wav
epochs=100
batch_size=32
lr=0.0001
n_save=5

python train.py -s $source_path -t $target_path -e $epochs -b $batch_size -l $lr -n $n_save 