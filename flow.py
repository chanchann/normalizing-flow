import tensorflow as tf
import numpy as np
import pandas as pd
import tensorflow_probability as tfp

tfd = tfp.distributions
tfb = tfp.bijectors
print("TFP Version", tfp.__version__)
print("TF  Version",tf.__version__)
np.random.seed(42)
tf.random.set_seed(42)

class MAF(tf.keras.models.Model):
    def __init__(self, *, output_dim, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.shift_and_log_scale_fn = tfb.AutoregressiveNetwork(
            params=2, hidden_units=[128, 128])
        # Defining the bijector
        num_bijectors = 5
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.MaskedAutoregressiveFlow(shift_and_log_scale_fn=self.shift_and_log_scale_fn))
            bijectors.append(tfb.BatchNormalization())
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.Normal(loc=0., scale=1.),
            bijector=bijector,
            event_shape=[self.output_dim])

    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)

class IAF(tf.keras.models.Model):

    def __init__(self, *, output_dim, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.shift_and_log_scale_fn= tfb.AutoregressiveNetwork(
            params=2, hidden_units=[512, 512])
        # Defining the bijector
        num_bijectors = 5
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.Invert(tfb.MaskedAutoregressiveFlow(
                                shift_and_log_scale_fn=self.shift_and_log_scale_fn)))
            bijectors.append(tfb.BatchNormalization())
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)

class MADE(tf.keras.models.Model):

    def __init__(self, *, output_dim, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.made = tfb.AutoregressiveNetwork(params=1, hidden_units=[32])
        # Defining the bijector
        num_bijectors = 5
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.MaskedAutoregressiveFlow(
                    lambda y: (self.made(y)[..., 0], None),
                    is_constant_jacobian=True))
            bijectors.append(tfb.BatchNormalization())
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)

class RNVP(tf.keras.models.Model):

    def __init__(self, *, output_dim, num_masked = 1, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked
        self.shift_and_log_scale_fn= tfb.real_nvp_default_template(
                            hidden_layers=[128, 128])
        # Defining the bijector
        num_bijectors = 6
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.RealNVP(num_masked=self.num_masked,shift_and_log_scale_fn=self.shift_and_log_scale_fn))
            bijectors.append(tfb.BatchNormalization())
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)

class NICE(tf.keras.models.Model):

    def __init__(self, *, output_dim, num_masked = 1, **kwargs): #** additional arguments for the super class
        super().__init__(**kwargs)
        self.output_dim = output_dim
        self.num_masked = num_masked
        self.shift_and_log_scale_fn= tfb.real_nvp_default_template(shift_only= True,
                            hidden_layers=[128, 128])
        # Defining the bijector
        num_bijectors = 5
        bijectors=[]
        for i in range(num_bijectors):
            bijectors.append(tfb.RealNVP(num_masked=self.num_masked,shift_and_log_scale_fn=self.shift_and_log_scale_fn,
                                         is_constant_jacobian=False))
            bijectors.append(tfb.BatchNormalization())
            bijectors.append(tfb.Permute(permutation=[1, 0]))
        # Discard the last Permute layer.
        bijector = tfb.Chain(list(reversed(bijectors[:-1])))
        
        # Defining the flow
        self.flow = tfd.TransformedDistribution(
            distribution=tfd.MultivariateNormalDiag(loc=[0., 0.]),
            bijector=bijector)

    def call(self, *inputs): 
        return self.flow.bijector.forward(*inputs)
    
    def getFlow(self, num):
        return self.flow.sample(num)

