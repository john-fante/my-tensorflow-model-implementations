"""
UNet Tensorflow implementation
(paper -> https://arxiv.org/abs/1505.04597 )

my accounts

- github -> https://github.com/john-fante
- kaggle -> https://www.kaggle.com/banddaniel
- stackoverflow -> https://stackoverflow.com/users/22880135/arturo-bandini-jr

"""


from tensorflow.keras.layers import Layer, Conv2D, Dropout, UpSampling2D, concatenate
from tensorflow.keras.layers import Add, Multiply, Input, MaxPool2D, BatchNormalization, Activation
from tensorflow.keras.models import Model
from keras import backend as K

#img height and width
IMG_SIZE = 400, 400


# Encoding block for U-net architecture
class EncoderBlock(Layer):

  def __init__(self, filters, kernel_size, rate, pooling = True, **kwargs):
    super(EncoderBlock, self).__init__(**kwargs)

    self.filters = filters
    self.kernel_size = kernel_size
    self.rate = rate
    self.pooling = pooling

    self.c1 = Conv2D(filters, kernel_size, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'lecun_normal')
    self.drop = Dropout(rate)
    self.c2 = Conv2D(filters, kernel_size, strides = 1, padding = 'same', activation = 'relu', kernel_initializer = 'lecun_normal')
    self.pool = MaxPool2D()


  def call(self, inputs):
    L = self.c1(inputs)
    L = self.drop(L)
    L = self.c2(L)
    if self.pooling:
      P = self.pool(L)
      return P, L
    else:
      return L


  def get_config(self):
    base_config = super().get_config()

    return {
          **base_config,
          "filters" : self.filters,
          "kernel_size": self.kernel_size,
          "rate" : self.rate,
          "pooling" : self.pooling
      }




# Decoding block for U-net architecture
class DecoderBlock(Layer):

  def __init__(self, filters, kernel_size, rate, **kwargs ):
    super(DecoderBlock, self).__init__(**kwargs)

    self.filters = filters
    self.kernel_size = kernel_size
    self.rate = rate

    self.up = UpSampling2D()
    self.net = EncoderBlock(filters, kernel_size, rate, pooling = False)


  def call(self, inputs):
    inputs, skip_inputs = inputs
    L = self.up(inputs)
    C_ = concatenate([L, skip_inputs ])
    L = self.net(C_)
    return L


  def get_config(self):
    base_config = super().get_config()
    return {
        **base_config,
        "filters" : self.filters,
        "kernel_size": self.kernel_size,
        "rate" : self.rate
    }


# channel 3 for RGB images
inp = Input(shape = (*IMG_SIZE, 3))

p1, c1 = EncoderBlock(32, 2, 0.1, name = 'Encoder1')(inp)
p2, c2 = EncoderBlock(64, 2, 0.1, name = 'Encoder2')(p1)
p3, c3 = EncoderBlock(128, 2, 0.2, name = 'Encoder3')(p2)
p4, c4 = EncoderBlock(256, 2, 0.2, name = 'Encoder4')(p3)

encoding = EncoderBlock(512 , 2, 0.3, pooling = False ,name = 'Encoding')(p4)

d1 = DecoderBlock(256, 2, 0.2 ,name = 'Decoder1' )([encoding, c4])
d2 = DecoderBlock(128 ,2, 0.2 ,name = 'Decoder2' )([d1, c3])
d3 = DecoderBlock(64 ,2, 0.1 ,name = 'Decoder3' )([d2, c2])
d4 = DecoderBlock(32 ,2, 0.1 ,name = 'Decoder4' )([d3, c1])

out = Conv2D(1, kernel_size = 1 ,activation ='sigmoid', padding = 'same')(d4)
model = Model(inputs = inp, outputs = out)
