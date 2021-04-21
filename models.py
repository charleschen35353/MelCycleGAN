import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import TruncatedNormal, he_normal
from layers import *
from utils import *


#U-NET style architecture
def build_generator(input_shape):
  h,w,c = input_shape
  inp = Input(shape=input_shape)
  #downscaling
  g0 = tf.keras.layers.ZeroPadding2D((0,1))(inp)
  g1 = conv2d(g0, 256, kernel_size=(h,3), strides=1, padding='valid')
  g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2))
  g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2))
  
  #upscaling
  g4 = deconv2d(g3,g2, 256, kernel_size=(1,7), strides=(1,2))
  g5 = deconv2d(g4,g1, 256, kernel_size=(1,9), strides=(1,2), bnorm=False)
  g6 = ConvSN2DTranspose(1, kernel_size=(h,1), strides=(1,1), kernel_initializer=tf.keras.initializers.he_uniform(), padding='valid', activation='tanh')(g5)
  return Model(inp,g6, name='G')

#Siamese Network
def build_siamese(input_shape):
  h,w,c = input_shape
  inp = Input(shape=input_shape)
  g1 = conv2d(inp, 256, kernel_size=(h,3), strides=1, padding='valid', sn=False)
  g2 = conv2d(g1, 256, kernel_size=(1,9), strides=(1,2), sn=False)
  g3 = conv2d(g2, 256, kernel_size=(1,7), strides=(1,2), sn=False)
  g4 = Flatten()(g3)
  g5 = Dense(vec_len)(g4)
  return Model(inp, g5, name='S')

#Discriminator (Critic) Network
def build_critic(input_shape):
  h,w,c = input_shape
  inp = Input(shape=input_shape)
  g1 = conv2d(inp, 512, kernel_size=(h,3), strides=1, padding='valid', bnorm=False)
  g2 = conv2d(g1, 512, kernel_size=(1,9), strides=(1,2), bnorm=False)
  g3 = conv2d(g2, 512, kernel_size=(1,7), strides=(1,2), bnorm=False)
  g4 = Flatten()(g3)
  g4 = DenseSN(1, kernel_initializer=tf.keras.initializers.he_uniform())(g4)
  return Model(inp, g4, name='C')
  
#Load past models from path to resume training or test

def load(path):
  agen = build_generator((hop,shape,1))
  bgen = build_generator((hop,shape,1))
  siam = build_siamese((hop,shape,1))
  acritic = build_critic((hop,3*shape,1))
  bcritic = build_critic((hop,3*shape,1))
  agen.load_weights(path+'/agen.h5')
  acritic.load_weights(path+'/acritic.h5')
  bgen.load_weights(path+'/bgen.h5')
  bcritic.load_weights(path+'/bcritic.h5')
  siam.load_weights(path+'/siam.h5')
  return agen,acritic,bgen,bcritic,siam

#Build models
def build():
  agen = build_generator((hop,shape,1))
  bgen = build_generator((hop,shape,1))
  siam = build_siamese((hop,shape,1))
  acritic = build_critic((hop,3*shape,1))
  bcritic = build_critic((hop,3*shape,1))
  return agen,acritic,bgen,bcritic,siam
  
  
#Get models and optimizers
def get_networks(shape, load_model=False, path=None, lr = 0.0001):
  if not load_model:
    agen,acritic,bgen,bcritic,siam = build()
    print('Built networks')
  else:
    agen,acritic,bgen,bcritic,siam = load(path)
    print('Model loaded')


  opt_gena = Adam(lr, 0.5)
  opt_disca = Adam(lr, 0.5)
  opt_genb = Adam(lr, 0.5)
  opt_discb = Adam(lr, 0.5)

  return agen,acritic,bgen,bcritic,siam, [opt_gena,opt_disca,opt_genb,opt_discb]

#Set learning rate
def update_lr(lr, opt):
  opt_gena.learning_rate = lr

