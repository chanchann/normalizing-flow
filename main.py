import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
tfb = tfp.bijectors
from time import time
import argparse

from flow import *
from generate_data import data_moon

parser = argparse.ArgumentParser()
parser.add_argument("--flow_type",type=str,default='MAF',dest='flow_type',choices = ['MAF','IAF','MADE','RNVP', 'NICE'],help="Flow types")
args = parser.parse_args()

np_samples = data_moon(2000)
print("data shape : " ,np_samples.shape)

if args.flow_type == 'MAF':
    model = MAF(output_dim=2)
elif args.flow_type == 'IAF':
    model = IAF(output_dim=2)
elif args.flow_type == 'MADE':
    model = MADE(output_dim=2)
elif args.flow_type == 'RNVP':
    model = RNVP(output_dim=2)
elif args.flow_type == 'NICE':
    model = NICE(output_dim=2)
else:
    print("Not implement yet")

print('Flow type:', args.flow_type)

_ = model(np_samples) 

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

@tf.function #Adding the tf.function makes it about 10 times faster!!!
def train_step(X): 
    with tf.GradientTape() as tape:
        loss = -tf.reduce_mean(model.flow.log_prob(X)) 
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss      

start = time()
for i in range(101):
    loss = train_step(np_samples)
    if (i % 50 == 0):
        print(i, " ",loss.numpy(), (time()-start))
        start = time()