import numpy as np
import tensorflow as tf
import os
import sys
import csv
import math

def md_mse_std_complex_loss(y_pred, y):
  mean_diff = tf.reduce_mean(y_pred, axis=0) - tf.reduce_mean(tf.cast(y, dtype=y_pred.dtype), axis=0)
  mean_loss = tf.reduce_mean(tf.reduce_sum(tf.cast(mean_diff*tf.math.conj(mean_diff), dtype=tf.float64), axis=1))

  std_diff = tf.math.reduce_std(y_pred, axis=0) - tf.math.reduce_std(tf.cast(y, dtype=y_pred.dtype), axis=0)
  std_loss = tf.reduce_mean(tf.reduce_sum(std_diff*tf.math.conj(std_diff), axis=1))

  assert(not math.isnan(mean_loss))
  assert(not math.isnan(std_loss))

  return mean_loss + std_loss

def md_mse_std_loss(y_pred, y):
  mean_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.reduce_mean(y_pred, axis=0) - tf.reduce_mean(tf.cast(y, dtype=tf.float32), axis=0)), axis=1))
  std_loss = tf.reduce_mean(tf.reduce_sum(tf.square(tf.math.reduce_std(y_pred, axis=0) - tf.math.reduce_std(tf.cast(y, dtype=tf.float32), axis=0)), axis=1))

  return mean_loss + std_loss

def mse_mean_loss(y_pred, y):
  return tf.reduce_mean(tf.square(tf.reduce_mean(y_pred, axis=0) - tf.reduce_mean(tf.cast(y, dtype=tf.float32), axis=0)))

def mse_std_loss(y_pred, y):
  mean_loss = tf.reduce_mean(tf.square(tf.reduce_mean(y_pred, axis=0) - tf.reduce_mean(tf.cast(y, dtype=tf.float32), axis=0)))
  std_loss = tf.reduce_mean(tf.square(tf.math.reduce_std(y_pred, axis=0) - tf.math.reduce_std(tf.cast(y, dtype=tf.float32), axis=0)))

  return mean_loss + std_loss

def mse_var_loss(y_pred, y):
  mean_loss = tf.reduce_mean(tf.square(tf.reduce_mean(y_pred, axis=0) - tf.reduce_mean(tf.cast(y, dtype=tf.float32), axis=0)))
  var_loss = tf.reduce_mean(tf.square(tf.square(tf.math.reduce_std(y_pred, axis=0)) - tf.square(tf.math.reduce_std(tf.cast(y, dtype=tf.float32), axis=0))))

  return mean_loss + var_loss

def mse_loss(y_pred, y):
  return tf.reduce_mean(tf.reduce_mean(tf.square(y_pred - y), axis=0))

def mse_mean_std_loss(y_pred, y):
  mean_loss = tf.reduce_mean(tf.reduce_mean(tf.square(y_pred - y), axis=0))
  std_loss = tf.reduce_mean(tf.math.reduce_std(tf.square(y_pred - y), axis=0))

  return mean_loss + std_loss

def abs_err(y_pred, y, batch_size):
  m = tf.shape(y_pred)[0]/batch_size
  batch_means = tf.reduce_mean(tf.reshape(tf.math.abs(y_pred - y), [tf.cast(m, dtype=tf.int32), -1, tf.shape(y_pred)[1]]), axis=1)
  err = tf.reduce_mean(batch_means, axis=0)
  std = tf.math.reduce_std(batch_means, axis=0)

  return err, std, std/tf.cast(tf.math.sqrt(m), dtype=tf.float32)

def abs_err_all(y_pred, y):
  mean = tf.reduce_mean(tf.math.abs(y_pred - y)[:,-1])
  sigma = tf.math.reduce_std(tf.math.abs(y_pred - y)[:,-1])
  return mean, sigma, sigma/tf.math.sqrt(tf.cast(tf.shape(y_pred)[0], dtype=tf.float32))

def abs_err_loss(y_pred, y):
  return tf.reduce_mean(tf.math.abs(y_pred - y)[:,-1])

def mean_abs_err_loss(y_pred, y):
  return tf.reduce_mean(tf.reduce_mean(tf.math.abs(y_pred - y), axis=0))

def fit_model(x0, y, sde_mod, loss_func, batch_size, epochs=10, learning_rate=0.01):
  num_traj = y.shape[0]

  y0 = sde_mod(x0, num_traj)

  x0vec = tf.ones([y.shape[0],1], dtype=x0.dtype)*x0

  init_loss = loss_func(y0, y)
  print('Init loss:', init_loss)

  dataset = tf.data.Dataset.from_tensor_slices((x0vec, tf.cast(y, dtype=x0.dtype)))
  dataset = dataset.shuffle(buffer_size=x0vec.shape[0]).batch(batch_size)

  losses = [init_loss]

  for epoch in range(epochs):
    for x_batch, y_batch in dataset:
      # Evaluate the loss function
      with tf.GradientTape() as tape:
        batch_loss = loss_func(sde_mod(x_batch), y_batch)
      
      # Calculate the gradient and update
      grads = tape.gradient(batch_loss, sde_mod.variables)
      for g, v in zip(grads, sde_mod.variables):
        if g is not None:
          v.assign_sub(learning_rate*g)
    
    # Calculate the loss for this epoch
    loss = loss_func(sde_mod(x0vec), y)
    losses.append(loss)

    if epoch % 1 == 0:
      print(f'Loss for epoch {epoch} = {loss.numpy():0.3f}')
      print(sde_mod.variables)

  return losses

class EulerMultiDModel(tf.Module):

  def __init__(self, mint, maxt, deltat, a, b, d, m, num_params, params=None, fix_params=None):
    self.num_params = num_params
    self.params = []
    for pidx in range(self.num_params):
      # Randomly generate model parameters for those not provided
      if params is None or params[pidx] is None:
        self.params.append(tf.Variable(np.random.uniform()))
      else:
        self.params.append(tf.Variable(params[pidx], trainable=(fix_params is not None and not fix_params[pidx])))

    #self.tvec = tf.range(mint,maxt,deltat)
    self.tvec = np.arange(mint,maxt,deltat)
    self.deltat = deltat

    self.a = a
    self.b = b

    self.d = d
    self.m = m
  
  #@tf.function
  def __call__(self, x0, num_traj=None, wvec=None):
    if num_traj is None:
      num_traj = tf.shape(x0)[0]

    if wvec is not None:
      self.wvec = wvec
    else:
      self.wvec = tf.cast(tf.random.normal(stddev=math.sqrt(self.deltat), shape=[num_traj,self.tvec.shape[0]-1,self.m,1]), dtype=x0.dtype)

    prevy = tf.ones(shape=[num_traj,self.d,1], dtype=x0.dtype)*tf.reshape(x0,[-1,self.d,1])
    y = tf.reshape(prevy, [num_traj,self.d,1])

    for tidx, t in enumerate(self.tvec[:-1]):
      curry = prevy + self.a(t,prevy,self.params)*self.deltat + tf.matmul(self.b(t,prevy,self.params),self.wvec[:,tidx,:,:])
      y = tf.concat([y, curry], axis=2)
      prevy = curry

    return y

def getrhop(p):
  rhop = 0.0
  for r in range(1,p+1):
    rhop = rhop + 1/float(r*r)
  rhop = (1/12.0) - rhop/(2*math.pi**2)
  return rhop

def multiintj12(m,p,deltat,wvec):
  '''
  Returns:
  jmat - [num_traj,num_times-1,m,m]
  '''

  num_traj = wvec.shape[0]
  num_times = wvec.shape[-1] + 1

  gsi = wvec/math.sqrt(deltat)
  mu = tf.random.normal(stddev=1.0, shape=[num_traj,num_times-1,m,1])
  eta = tf.random.normal(stddev=1.0, shape=[num_traj,num_times-1,m,1,p])
  zeta = tf.random.normal(stddev=1.0, shape=[num_traj,num_times-1,m,1,p])

  sumval = tf.zeros([num_traj,num_times-1,m,m])
  rhop = 0.0
  for ridx in range(p):
    r = float(ridx + 1)
    sumval = sumval + (1/r)*(tf.matmul(zeta[:,:,:,:,ridx], tf.transpose(math.sqrt(2)*gsi + eta[...,ridx], perm=[0,1,3,2])))
    rhop = rhop + (1/(r*r))

  sumval = (sumval - tf.transpose(sumval, perm=[0,1,3,2]))
  rhop = (1/12.0) - rhop/(2*math.pi**2)
  jmat = sumval*deltat/math.pi
  jmat = jmat + deltat*(0.5*tf.matmul(gsi, tf.transpose(gsi, perm=[0,1,3,2])) + \
                        math.sqrt(rhop)*(tf.matmul(mu, tf.transpose(gsi, perm=[0,1,3,2])) - tf.matmul(gsi, tf.transpose(mu, perm=[0,1,3,2]))))
  jmat = jmat - jmat*tf.eye(m,m,[num_traj,num_times-1]) # Zero out the diagonal
  return jmat

class MilsteinModel(tf.Module):

  def __init__(self, mint, maxt, deltat, a, b, bp, d, m, p, num_params, params=None, fix_params=None):
    '''
    Input functions defining the SDE:
    a - Returns [num_traj,d,1] drift function of t and x([num_traj,d,1])
    b - Returns [num_traj,d,m] diffusion functions of t and x([num_traj,d,1])
    bp - Returns [num_traj,m,d,d] Jacobians of b_j w.r.t. x (shape [d,d]), where the second index indicates
         which column of b is being differentiated. Is a function of t and x([num_traj,d,1])
    '''
    self.num_params = num_params
    self.params = []
    for pidx in range(self.num_params):
      # Randomly generate model parameters for those not provided
      if params is None or params[pidx] is None:
        self.params.append(tf.Variable(np.random.uniform()))
      else:
        self.params.append(tf.Variable(params[pidx], trainable=(fix_params is not None and not fix_params[pidx])))

    #self.tvec = tf.range(mint,maxt,deltat)
    self.tvec = np.arange(mint,maxt,deltat)
    self.deltat = deltat

    self.a = a
    self.b = b
    self.bp = bp

    self.d = d
    self.m = m

    self.p = p
  
  #@tf.function
  def __call__(self, x0, num_traj=None, wvec=None):
    if num_traj is None:
      num_traj = tf.shape(x0)[0]
    
    num_times = self.tvec.shape[0]

    if wvec is not None:
        self.wvec = wvec
    else:
        self.wvec = tf.cast(tf.random.normal(stddev=math.sqrt(self.deltat), shape=[num_traj,num_times-1,self.m,1]), dtype=x0.dtype)

    self.jmat = multiintj12(self.m, self.p, self.deltat, self.wvec) # [num_traj,num_times-1,m,m]
    self.imat = 0.5*tf.eye(self.m, self.m, [num_traj,num_times-1])*(tf.matmul(self.wvec, tf.transpose(self.wvec, perm=[0,1,3,2])) - self.deltat) + self.jmat # [num_traj,num_times-1,m,m]

    prevy = tf.ones(shape=[num_traj,self.d,1], dtype=x0.dtype)*tf.reshape(x0,[-1,self.d,1])
    y = tf.reshape(prevy, [num_traj,self.d,1])

    for tidx, t in enumerate(self.tvec[:-1]):
      # y_n+1 = y_n + Drift \ Euler \ Milstein
      bb = self.b(t,prevy,self.params) # [num_traj,d,m]
      lb = tf.matmul(tf.reshape(tf.transpose(bb, perm=[0,2,1]), [num_traj,1,self.m,self.d]), self.bp(t,prevy,self.params)) # [num_traj,m,m,d], indices = [i,j2,j1,k]
      lb = tf.transpose(lb, perm=[0,3,2,1]) # [num_traj,d,m,m], indices = [i,k,j1,j2]
      
      curry = prevy + self.a(t,prevy,self.params)*self.deltat + \
              tf.matmul(bb,self.wvec[:,tidx,:,:]) + \
              tf.expand_dims(tf.reduce_sum(lb*tf.expand_dims(self.imat[:,tidx,:,:], 1), axis=[2,3]), -1)
      y = tf.concat([y, curry], axis=2)
      prevy = curry

    return y

class EulerModel(tf.Module):

  def __init__(self, mint, maxt, deltat, a, b, num_params, params=None, fix_params=None):
    self.num_params = num_params
    self.params = []
    for pidx in range(self.num_params):
      # Randomly generate model parameters for those not provided
      if params is None or params[pidx] is None:
        self.params.append(tf.Variable(np.random.uniform()))
      else:
        self.params.append(tf.Variable(params[pidx], trainable=(fix_params is not None and not fix_params[pidx])))

    #self.tvec = tf.range(mint,maxt,deltat)
    self.tvec = np.arange(mint,maxt,deltat)
    self.deltat = deltat

    self.a = a
    self.b = b
  
  #@tf.function
  def __call__(self, x0, num_traj=None):
    if num_traj is None:
      num_traj = tf.shape(x0)[0]
    self.wvec = tf.random.normal(stddev=math.sqrt(self.deltat), shape=[num_traj,self.tvec.shape[0]-1])

    y = tf.ones(shape=[num_traj,1])*x0
    prevy = y

    for tidx, t in enumerate(self.tvec[:-1]):
      curry = prevy + self.a(t,prevy,self.params)*self.deltat + self.b(t,prevy,self.params)*tf.reshape(self.wvec[:,tidx], [num_traj,1])
      y = tf.concat([y, curry],1)
      prevy = curry

    return y

def euler_sde_np(x0,a,b,mint,maxt,deltat,num_traj=1):
  '''euler_sde_np: An SDE solver using the Euler scheme implemented with numpy.
                   Solves the SDE dX_t = a(t,X_t)dt + b(t,X_t)dW_t

  Args:
  x0: Initial value of the random process
  a: The drift function a(t,x)
  b: The diffusion function b(t,x)
  mint: Start time
  maxt: End time
  deltat: time step
  num_traj: Number of trajectories to simulate

  Returns:
  tvec: Time vector [mint:maxt:deltat], shape = [num_times]
  yvec: Simulated solutions, shape = [num_traj,num_times]
  wvec: Gaussian process samples used to simulate yvec, shape = [num_traj,num_times-1]
  '''

  tvec = np.arange(mint,maxt,deltat)
  yvec = np.zeros([num_traj,tvec.shape[0]])
  wvec = np.random.default_rng().normal(scale=math.sqrt(deltat), size=[num_traj,tvec.shape[0]-1])

  yvec[:,0] = x0
  for tidx, t in enumerate(tvec[:-1]):
    yvec[:,tidx+1] = yvec[:,tidx] + a(t,yvec[:,tidx])*deltat + b(t,yvec[:,tidx])*wvec[:,tidx]

  return tvec, yvec, wvec

def euler_sde_tf(x0,a,b,mint,maxt,deltat,num_traj=1):
  '''euler_sde_tf: An SDE solver using the Euler scheme implemented with TensorFlow.
                   Solves the SDE dX_t = a(t,X_t)dt + b(t,X_t)dW_t

  Args:
  x0: Initial value of the random process, shape = [None,1]
  a: The drift function a(t,x)
  b: The diffusion function b(t,x)
  mint: Start time
  maxt: End time
  deltat: time step
  num_traj: Number of trajectories to simulate

  Returns:
  tvec: Time vector [mint:maxt:deltat], shape = [num_times]
  yvec: Simulated solutions, shape = [None,num_times]
  wvec: Gaussian process samples used to simulate yvec, shape = [None,num_times-1]
  '''

  tvec = np.arange(mint,maxt,deltat)
  yvec = np.zeros([num_traj,tvec.shape[0]])
  wvec = np.zeros([num_traj,tvec.shape[0]-1])

  prevy = tf.ones(num_traj)*x0
  yvec[:,0] = prevy.numpy()

  for tidx, t in enumerate(tvec[:-1]):
    dw = tf.random.normal(stddev=math.sqrt(deltat), shape=[num_traj])
    curry = prevy + a(t,prevy)*deltat + b(t,prevy)*dw
    yvec[:,tidx+1] = curry.numpy()
    wvec[:,tidx] = dw.numpy()
    prevy = curry

  return tvec, yvec, wvec