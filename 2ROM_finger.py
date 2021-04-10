import numpy as np
import cupy as cp
import math
import random
import matplotlib.pyplot as plt
import timeit
import multiprocessing as mp
import functools
import operator

from math import sin,cos,radians
from mpl_toolkits.mplot3d import Axes3D
from tqdm import tqdm
from itertools import combinations, product
from joblib import Parallel, delayed
from numba import jit, prange
from cupy import prof

# GPU Accelerated Point Cloud Code
@jit(nopython=True)
def MCP(t1,t2,t3,a0):
  r1 = radians(t1)
  r2 = radians(t2)
  r3 = radians(t3)
  s1 = sin(r1)
  c1 = cos(r1)
  s2 = sin(r2)
  c2 = cos(r2)
  s3 = sin(r3)
  c3 = cos(r3)
  return [a0*c2*c1, a0*s2*c1, -a0*s1]

@jit(nopython=True)
def PIP(t1,t2,t3,t4,t5,a0,a1):
  r1 = radians(t1)
  r2 = radians(t2)
  r3 = radians(t3)
  r4 = radians(t4)
  r5 = radians(t5)
  s1 = sin(r1)
  c1 = cos(r1)
  s2 = sin(r2)
  c2 = cos(r2)
  s3 = sin(r3)
  c3 = cos(r3)
  s4 = sin(r4)
  c4 = cos(r4)
  s5 = sin(r5)
  c5 = cos(r5)
  m00 = c2*c1*c4-s2*c3*s4+c2*s1*s3*s4
  m01 = -c2*c1*s4-s2*c3*c4+c2*s1*s3*c4
  m02 = s2*s3+c2*s1*c3
  m03 = a0*c2*c1
  m10 = s2*c1*c4+c2*c3*s4+s2*s1*s3*s4
  m11 = -s2*c1*s4+c2*c3*c4+s2*s1*s3*c4
  m12 = -c2*s3+s2*s1*c3
  m13 = a0*s2*c1
  m20 = -s1*c4+c1*s3*s4
  m21 = s1*s4+c1*s3*c4
  m22 = c1*c3
  m23 = -a0*s1

  return ((m00*c5-m02*s5, m01, m00*s5+m02*c5),
          (m10*c5-m12*s5, m11, m10*s5+m12*c5),
          (m20*c5-m22*s5, m21, m20*s5+m22*c5)), [m03+a1*(m00*c5-m02*s5), m13+a1*(m10*c5-m12*s5), m23+a1*(m20*c5-m22*s5)]

@jit(nopython=True, parallel=True)
def compute(all_t, a0, a1):
  out = [(((0.0,0.0,0.0),(0.0,0.0,0.0),(0.0,0.0,0.0)), [0.0,0.0,0.0], [0.0,0.0,0.0])]*len(all_t)
  for i in range(len(all_t)):
    t1,t2,t3,t4,t5 = all_t[i]
    t6 = t3
    loc_MCP_rel_wrist = MCP(t1,t2,t3,a0)
    key, loc_PIP_rel_wrist = PIP(t1,t2,t3,t4,t5,a0,a1)
    out[i] = (key,loc_MCP_rel_wrist,loc_PIP_rel_wrist)
    if i%500000 == 0:
      print(i)
  return out

def print_pc_metadata(a0,a1,t1_r,t2_r,t3_r,t4_r,t5_r,n):
  SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
  print('--------------------------------------------------------')
  print('| a0: {:<20}\ta1: {:<19} |'.format(a0,a1))
  print('| {:<5} : {:<5} to {:<5} with step {:<5} (total: {:<5}) |'.format('\N{GREEK CAPITAL LETTER THETA}1'.translate(SUB),t1_r[0],t1_r[-1],1,len(t1_r)))
  print('| {:<5} : {:<5} to {:<5} with step {:<5} (total: {:<5}) |'.format('\N{GREEK CAPITAL LETTER THETA}2'.translate(SUB),t4_r[0],t2_r[-1],1,len(t2_r)))
  print('| {:<5} : {:<5} to {:<5} with step {:<5} (total: {:<5}) |'.format('\N{GREEK CAPITAL LETTER THETA}3'.translate(SUB),t3_r[0],t3_r[-1],1,len(t3_r)))
  print('| {:<5} : {:<5} to {:<5} with step {:<5} (total: {:<5}) |'.format('\N{GREEK CAPITAL LETTER THETA}4'.translate(SUB),t4_r[0],t4_r[-1],1,len(t4_r)))
  print('| {:<5} : {:<5} to {:<5} with step {:<5} (total: {:<5}) |'.format('\N{GREEK CAPITAL LETTER THETA}5'.translate(SUB),t5_r[0],t5_r[-1],1,len(t5_r)))
  print('| total points: {:<19}'.format(n))
  print('--------------------------------------------------------')

def print_proc_metadata():
  num_cores = mp.cpu_count()
  mempool = cp.get_default_memory_pool()
  
  print('--------------------------------------------------------')
  print('| num_cpu_cores: {:<37} |'.format(num_cores))
  print('| mempool used bytes: {:<32} |'.format(mempool.used_bytes()))
  print('| mempool total bytes: {:<31} |'.format(mempool.total_bytes()))
  print('| mempool limit bytes: {:<31} |'.format(cp.get_default_memory_pool().get_limit()))
  print('--------------------------------------------------------')

def nth_product(n, *iterables):
    sizes = [len(iterable) for iterable in iterables]
    indices = [
        int((n/functools.reduce(operator.mul, sizes[i+1:], 1)) % sizes[i])
        for i in range(len(sizes))]
    return tuple(iterables[i][idx] for i, idx in enumerate(indices))

def compute_blocks(block_size,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn):
  j = int(num_points / block_size)
  for i in range(j):
    some_t = [nth_product(n,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn) for n in range(i*block_size,min(num_points,(i+1)*block_size))]
    all_t_gpu = cp.array(some_t)
    #all_t_gpu = cp.array(cp.meshgrid(t1_r, t2_r, t3_r, t4_r, t5_r, copy=False)).T.reshape(-1,5)

    #print(all_t_gpu[:5,:])
    #print("Finished creating the grid")

    results = compute(cp.asnumpy(all_t_gpu), a0, a1)

if __name__ == '__main__':
  MCP_point_cloud = {}
  PIP_point_cloud = {}

  count = 0
  stop = 10000000

  t1_rn = np.arange(-60,61,1)
  t2_rn = np.arange(-30,21,1)
  t3_rn = np.arange(0,181,1)
  t4_rn = np.arange(-20,26,1)
  t5_rn = np.arange(-120,121,1)

  # Point Cloud Input Parameters
  a0 = 3
  a1 = 1
  t1_r = cp.arange(-60,61,1,dtype='int16')
  t2_r = cp.arange(-30,21,1,dtype='int16')
  t3_r = cp.arange(0,181,1,dtype='int16')
  t4_r = cp.arange(-20,26,10,dtype='int16')
  t5_r = cp.arange(-120,121,10,dtype='int16')

  num_points = len(t1_r)*len(t2_r)*len(t3_r)*len(t4_r)*len(t5_r)

  print_pc_metadata(a0,a1,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn,num_points)
  print_proc_metadata()
  
  results = []
  for num in range(-60,61,1):
    all_t_gpu = cp.array(cp.meshgrid(cp.array([num]), t2_r, t3_r, t4_r, t5_r, copy=False)).T.reshape(-1,5)
    results.append(compute(cp.asnumpy(all_t_gpu), a0, a1))

  print(len(results[0]))
    #all_t_gpu = cp.array(cp.meshgrid(cp.array([-59]), t2_r, t3_r, t4_r, t5_r, copy=False)).T.reshape(-1,5)
    #results.append(compute(cp.asnumpy(all_t_gpu), a0, a1))

  #a = 0
  #some_t = [nth_product(n,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn) for n in range(a*3000000,min(num_points,(a+1)*3000000))]
  #print(type(some_t[:5][0]))
  #all_t_gpu = cp.array(some_t)
  #results = compute(cp.asnumpy(all_t_gpu), a0, a1)

  #a = 1
  #some_t = [nth_product(n,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn) for n in range(a*3000000,min(num_points,(a+1)*3000000))]
  #all_t_gpu = cp.array(some_t)
  #results = compute(cp.asnumpy(all_t_gpu), a0, a1)

  #compute_blocks(3000000,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn)

  #all_t = product(t1_rn, t2_rn, t3_rn, t4_rn, t5_rn)
  #j = int(num_points / 3000000)
  #for i in range(j):
  #  some_t = [nth_product(n,t1_rn,t2_rn,t3_rn,t4_rn,t5_rn) for n in range(i*3000000,min(num_points,(i+1)*3000000))]
  #  all_t_gpu = cp.array(some_t)
  #  #all_t_gpu = cp.array(cp.meshgrid(t1_r, t2_r, t3_r, t4_r, t5_r, copy=False)).T.reshape(-1,5)

    #print(all_t_gpu[:5,:])
    #print("Finished creating the grid")

   # results = compute(cp.asnumpy(all_t_gpu), a0, a1)
  #compute.parallel_diagnostics(level=4)
  print_proc_metadata()
  print("Finished computing")
