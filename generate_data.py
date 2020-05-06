import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_probability as tfp
tfd = tfp.distributions

def data_moon(sample_num):
    x2_dist = tfd.Normal(loc=0., scale=4.)
    x2_samples = x2_dist.sample(sample_num)
    x1 = tfd.Normal(loc=.25 * tf.square(x2_samples),
                    scale=tf.ones(sample_num, dtype=tf.float32))
    x1_samples = x1.sample()
    x_samples = tf.stack([x1_samples, x2_samples], axis=1)
    return x_samples.numpy()