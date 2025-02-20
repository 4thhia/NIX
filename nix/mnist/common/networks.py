from typing import Tuple, List

import jax
import jax.numpy as jnp
import flax.linen as nn


class EncoderModule(nn.Module):
    output_channels: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    pool_size: Tuple[int, int]

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.output_channels, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = nn.relu(x)
        x = nn.max_pool(x, window_shape=self.pool_size, strides=self.pool_size)
        return x
"""
# Encoder
(batch, 32, 32, 3)
# EncoderModule
(batch, 32, 32, 32)
(batch, 16, 16, 32)
# EncoderModule
(batch, 16, 16, 64)
(batch, 8, 8, 64)
# reshape
(batch, 8 * 8 * 64)
# Dense
(batch, 512)
# Dense
(batch, 16), (batch, 16)
"""
class Encoder(nn.Module):
    color_channels: int
    num_latent_features: int
    hidden_features: int = 512

    @nn.compact
    def __call__(self, x):
        h = EncoderModule(32, kernel_size=(3, 3), strides=(1, 1), pool_size=(2, 2))(x)
        h = EncoderModule(64, kernel_size=(3, 3), strides=(1, 1), pool_size=(2, 2))(h)

        h = h.reshape((h.shape[0], -1))
        h = nn.Dense(features=self.hidden_features)(h)
        mean = nn.Dense(features=self.num_latent_features)(h)
        logvar = nn.Dense(features=self.num_latent_features)(h)
        return mean, logvar

def sample_z(rng, mean, logvar):
    std = jnp.exp(0.5 * logvar)
    esp = jax.random.normal(rng, mean.shape)  # Random noise
    return mean + std * esp


class DecoderModule(nn.Module):
    output_channels: int
    scale: int
    kernel_size: Tuple[int, int]
    strides: Tuple[int, int]
    activation: str='relu'

    def setup(self):
        if self.activation == 'relu':
            self.activation_fn = nn.relu
        elif self.activation == 'sigmoid':
            self.activation_fn = nn.sigmoid
        else:
            self.activation_fn = lambda x: x

    @nn.compact
    def __call__(self, x):
        x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * self.scale, x.shape[2] * self.scale, x.shape[3]), method='nearest')
        x = nn.Conv(features=self.output_channels, kernel_size=self.kernel_size, strides=self.strides, padding="SAME")(x)
        x = self.activation_fn(x)
        return x
"""
# Decoder
(batch, 16)
# Dense
(batch, 8 * 8 * 64)
# reshape
(batch, 8, 8, 64)

# DecoderModule
(batch, 16, 16, 32)
# DecoderModule
(batch, 32, 32, 16)
# Conv
(batch, 32, 32, 1)
"""
class Decoder(nn.Module):
    color_channels: int
    decoder_input_size: int

    def setup(self):
        self.num_neurons_in_middle_layer = 64 * self.decoder_input_size**2

    @nn.compact
    def __call__(self, h):
        h = nn.Dense(features=self.num_neurons_in_middle_layer)(h)
        h = nn.relu(h)
        h = h.reshape((-1, self.decoder_input_size, self.decoder_input_size, 64))
        h = DecoderModule(output_channels=32, scale=2, kernel_size=(3, 3), strides=(1, 1))(h)
        h = DecoderModule(output_channels=16, scale=2, kernel_size=(3, 3), strides=(1, 1))(h)
        recon = DecoderModule(output_channels=self.color_channels, scale=1, kernel_size=(3, 3), strides=(1, 1), activation='sigmoid')(h)
        return recon


class Classifier(nn.Module):
    num_classes: int = 10

    @nn.compact
    def __call__(self, z):
        return nn.Dense(features=self.num_classes)(z) # for prediction: jnp.argmax(class_logits, axis=1)


class ConvBlock(nn.Module):
    output_channels: int

    @nn.compact
    def __call__(self, x):
        x = nn.Conv(features=self.output_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.output_channels, kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
        x = nn.relu(x)
        return x

class DownBlock(nn.Module):
    feature_unit: int
    blocks: int

    @nn.compact
    def __call__(self, x):
        skips = []
        for i in range(self.blocks):
            x = ConvBlock(self.feature_unit * (2**i))(x)
            skips.append(x)
            x = nn.max_pool(x, window_shape=(2, 2), strides=(2, 2))
        return x, skips

class UpBlock(nn.Module):
    feature_unit: int
    blocks: int

    @nn.compact
    def __call__(self, x, skips: List):
        for i in range(self.blocks-1, -1, -1):
            x = jax.image.resize(x, shape=(x.shape[0], x.shape[1] * 2, x.shape[2] * 2, x.shape[3]), method='nearest')
            x = nn.Conv(features=self.feature_unit * (2**i), kernel_size=(3, 3), strides=(1, 1), padding="SAME")(x)
            x = jnp.concatenate((x, skips.pop()), axis=-1)
            x = ConvBlock(self.feature_unit * (2**i))(x)
        return x

class WeightUNet(nn.Module):
    out_channels: int = 1
    feature_unit: int = 16
    blocks: int = 3
    activation: str='const'

    def setup(self):
        if self.activation == 'sigmoid':
            self.activation_fn = nn.sigmoid
        elif self.activation == 'tanh':
            self.activation_fn = nn.tanh
        else:
            self.activation_fn = lambda x: x

    @nn.compact
    def __call__(self, x):
        x, skips = DownBlock(self.feature_unit, self.blocks)(x)
        x = ConvBlock(self.feature_unit * (2**self.blocks))(x)
        x = UpBlock(self.feature_unit, self.blocks)(x, skips)

        # Final 1x1 Conv layer to produce output
        x = nn.Conv(features=self.out_channels, kernel_size=(1, 1))(x)
        x = self.activation_fn(x)

        return x