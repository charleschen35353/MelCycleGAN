#Training Loop
#Imports
import tensorflow as tf
import soundfile as sf
import numpy as np
import os
import time
from spec_utils import *
from utils import *
from losses import *
from models import *


@tf.function
def proc(x):
  return tf.image.random_crop(x, size=[hop, 3*shape, 1])


epochs = 25
batch_size = 128
lr = 0.0001
n_save = 5


agen,acritic,bgen,bcritic,siam, [opt_gena,opt_disca,opt_genb,opt_discb] = get_networks(shape, lr = 0.0001 , load_model=False, path=os.path.join(root_dir))

#American
awv = audio_array('./dataset/cmu_us_bdl_arctic/wav')                               #get waveform array from folder containing wav files
aspec = tospec(awv)                                                                 #get spectrogram array
adata = splitcut(aspec)                                                             #split spectrogams to fixed length
#Indian
bwv = audio_array('./dataset/cmu_us_ksp_arctic/wav')
bspec = tospec(bwv)
bdata = splitcut(bspec)

dsa = tf.data.Dataset.from_tensor_slices(adata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)

dsb = tf.data.Dataset.from_tensor_slices(bdata).repeat(50).map(proc, num_parallel_calls=tf.data.experimental.AUTOTUNE).shuffle(10000).batch(bs, drop_remainder=True)



@tf.function
def train_all(a,b):
  #splitting spectrogram in 3 parts
  aa,aa2,aa3 = extract_image(a)
  bb,bb2,bb3 = extract_image(b)

  with tf.GradientTape() as tape_gena, \
       tf.GradientTape() as tape_disca, \
       tf.GradientTape() as tape_genb, \
       tf.GradientTape() as tape_discb:

    ''' for genA'''
    #translating A to B
    fab = agen(aa, training=True)
    fab2 = agen(aa2, training=True)
    fab3 = agen(aa3, training=True)

    #reconstruct B to A
    faba = bgen(fab, training=True)
    faba2 = bgen(fab2, training=True)
    faba3 = bgen(fab3, training=True)

    #identity mapping B to B                                                        COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
    fidbb = agen(bb, training=True)
    fidbb2 = agen(bb2, training=True)
    fidbb3 = agen(bb3, training=True)

    ''' for genB'''
    #translating B to A
    fba = bgen(bb, training=True)
    fba2 = bgen(bb2, training=True)
    fba3 = bgen(bb3, training=True)

    #reconstruct A to B
    fbab = agen(fba, training=True)
    fbab2 = agen(fba2, training=True)
    fbab3 = agen(fba3, training=True)
    
    #identity mapping A to A                                                        COMMENT THESE 3 LINES IF THE IDENTITY LOSS TERM IS NOT NEEDED
    fidaa = bgen(aa, training=True)
    fidaa2 = bgen(aa2, training=True)
    fidaa3 = bgen(aa3, training=True)

    #concatenate/assemble converted spectrograms
    fabtot = assemble_image([fab,fab2,fab3])
    fbatot = assemble_image([fba,fba2,fba3])

    cab = acritic(fabtot, training=True)
    cb = acritic(b, training=True)

    cba = bcritic(fbatot, training=True)
    ca = bcritic(a, training=True)

    #feed 2 pairs (A,G(A)) extracted spectrograms to Siamese
    sab = siam(fab, training=True)
    sab2 = siam(fab3, training=True)
    sa = siam(aa, training=True)
    sa2 = siam(aa3, training=True)
    
    #feed 2 pairs (G(B),A) extracted spectrograms to Siamese
    sb = siam(bb, training=True)
    sb2 = siam(bb3, training=True)
    sba = siam(fba, training=True)
    sba2 = siam(fba3, training=True)

    #cyclic loss for a
    loss_cya = (mae(aa,faba)+mae(aa2,faba2)+mae(aa3,faba3))/3.
    #cylic loss for b
    loss_cyb = (mae(bb,fbab)+mae(bb2,fbab2)+mae(bb3,fbab3))/3.

    #generator and critic losses for a
    loss_ga = g_loss_f(cab)
    loss_dra = d_loss_r(cb)
    loss_dfa = d_loss_f(cab)
    loss_da = (loss_dra+loss_dfa)/2.

    #generator and critic losses for b
    loss_gb = g_loss_f(cba)
    loss_drb = d_loss_r(ca)
    loss_dfb = d_loss_f(cba)
    loss_db = (loss_drb+loss_dfb)/2.

    #identity mapping loss for gena
    loss_ida = (mae(bb,fidbb)+mae(bb2,fidbb2)+mae(bb3,fidbb3))/3.                         #loss_id = 0. IF THE IDENTITY LOSS TERM IS NOT NEEDED
    
    #identity mapping loss for genb
    loss_idb = (mae(aa,fidaa)+mae(aa2,fidaa2)+mae(aa3,fidaa3))/3
    
    #travel loss for siam on a
    loss_ma = loss_travel(sa,sab,sa2,sab2)+loss_siamese(sa,sa2)

    #travel loss for siam on b
    loss_mb = loss_travel(sb,sba,sb2,sba2)+loss_siamese(sb,sb2)
    
    #generator+siamese total loss for gena
    lossgatot = loss_ga+10.*loss_ma+ 10.*loss_cya +0.5*loss_ida                                      #CHANGE LOSS WEIGHTS HERE  (COMMENT OUT +w*loss_id IF THE IDENTITY LOSS TERM IS NOT NEEDED)
    
    #generator+siamese total loss for genb
    lossgbtot = loss_gb+10.*loss_mb+ 10.*loss_cyb +0.5*loss_idb

  #computing and applying gradients
  grad_gena = tape_gena.gradient(lossgatot, agen.trainable_variables+siam.trainable_variables)
  opt_gena.apply_gradients(zip(grad_gena, agen.trainable_variables+siam.trainable_variables))

  grad_disca = tape_disca.gradient(loss_da, acritic.trainable_variables)
  opt_disca.apply_gradients(zip(grad_disca, acritic.trainable_variables))

  grad_genb = tape_genb.gradient(lossgbtot, bgen.trainable_variables+siam.trainable_variables)
  opt_genb.apply_gradients(zip(grad_genb, bgen.trainable_variables+siam.trainable_variables))

  grad_discb = tape_discb.gradient(loss_db, bcritic.trainable_variables)
  opt_discb.apply_gradients(zip(grad_discb, bcritic.trainable_variables))
  
  return loss_dra,loss_dfa,loss_ga,loss_ida,loss_drb,loss_drb,loss_gb,loss_idb



#Save in training loop
def save_end(epoch,gloss,closs,mloss,n_save=3,save_path=root_dir):                 #use custom save_path (i.e. Drive '../content/drive/My Drive/')
  if epoch % n_save == 0:
    print('Saving...')
    path = f'{save_path}/MELGANVC-{str(gloss)[:9]}-{str(closs)[:9]}-{str(mloss)[:9]}'
    os.mkdir(path)
    agen.save_weights(path+'/agen.h5')
    acritic.save_weights(path+'/acritic.h5')
    bgen.save_weights(path+'/bgen.h5')
    bcritic.save_weights(path+'/bcritic.h5')
    siam.save_weights(path+'/siam.h5')


df_lista = []
dr_lista = []
g_lista= []
id_lista = []
df_listb = []
dr_listb = []
g_listb = []
id_listb = []
c = 0
g = 0

for epoch in range(epochs):
    bef = time.time()
    
    for batchi,(a,b) in enumerate(zip(dsa,dsb)):
      
        dloss_ta,dloss_fa,glossa,idlossa,dloss_tb,dloss_fb,glossb,idlossb= train_all(a,b)

        df_lista.append(dloss_fa)
        dr_lista.append(dloss_ta)
        g_lista.append(glossa)
        id_lista.append(idlossa)
        df_listb.append(dloss_fb)
        dr_listb.append(dloss_tb)
        g_listb.append(glossb)
        id_listb.append(idlossb)
        c += 1
        g += 1
        
        if batchi%600==0:
            print(f'[Epoch {epoch}/{epochs}] [Batch {batchi}]')
            print(f'[D_A loss f: {np.mean(df_lista[-g:], axis=0)}', end='')
            print(f'r: {np.mean(dr_lista[-g:], axis=0)}] ', end='')
            print(f'[G_A loss: {np.mean(g_lista[-g:], axis=0)}] ', end='')
            print(f'[ID_A loss: {np.mean(id_lista[-g:])}] ', end='')
            print(f'[LR: {lr}]')
            print("\n")
            print(f'[D_B loss f: {np.mean(df_listb[-g:], axis=0)}', end='')
            print(f'r: {np.mean(dr_listb[-g:], axis=0)}] ', end='')
            print(f'[G_B loss: {np.mean(g_listb[-g:], axis=0)}] ', end='')
            print(f'[ID_B loss: {np.mean(id_listb[-g:])}] ', end='')
            print(f'[LR: {lr}]')
            g = 0
        nbatch=batchi

    print(f'Time/Batch {(time.time()-bef)/max(1,nbatch)}')
    print(g_lista[-n_save*c:])
    save_end(epoch,np.mean(g_lista[-n_save*c:], axis=0),np.mean(df_lista[-n_save*c:], axis=0),np.mean(id_lista[-n_save*c:], axis=0),n_save=n_save)
    print(f'Mean D_A loss: {np.mean(df_lista[-c:], axis=0)} Mean G_A loss: {np.mean(g_lista[-c:], axis=0)} Mean ID_A loss: {np.mean(id_lista[-c:], axis=0)}')
    print(f'Mean D_B loss: {np.mean(df_listb[-c:], axis=0)} Mean G_B loss: {np.mean(g_listb[-c:], axis=0)} Mean ID_B loss: {np.mean(id_listb[-c:], axis=0)}')
    c = 0
                  
