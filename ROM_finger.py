import numpy as np
import math
from math import sin,cos,radians
import random
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import timeit
from tqdm import tqdm
from itertools import combinations, product
import multiprocessing
from joblib import Parallel, delayed
from numba import jit, prange
import cupy


def T_x(theta,a):
  return np.matrix([[1,0,0,a],
                    [0,math.cos(math.radians(theta)),-math.sin(math.radians(theta)),0],
                    [0,math.sin(math.radians(theta)),math.cos(math.radians(theta)),0],
                    [0,0,0,1]])
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

#  return ((c2*c1, -s2*c3+c2*s1*s3, s2*s3+c2*s1*c3),
#          (s2*c1, c2*c3+s2*s1*s3, -c2*s3+s2*s1*c3),
#          (-s1, c1*s3, c1*c3)), 
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

def T_y(theta):
  return np.matrix([[math.cos(math.radians(theta)),0,math.sin(math.radians(theta)),0],
                    [0,1,0,0],
                    [-math.sin(math.radians(theta)),0,math.cos(math.radians(theta)),0],
                    [0,0,0,1]])

def T_z(theta):
  return np.matrix([[math.cos(math.radians(theta)),-math.sin(math.radians(theta)),0,0],
                    [math.sin(math.radians(theta)),math.cos(math.radians(theta)),0,0],
                    [0,0,1,0],
                    [0,0,0,1]])

def MCP_rel_wrist(t1,t2,t3,a0):
  return np.matmul(np.matmul(T_z(t2),T_y(t1)),T_x(t3,a0))
#  V = np.matrix([[1],[1],[1],[1]])
#  return np.matmul(T,V)

def PIP_rel_wrist(t1,t2,t3,t4,t5,a0,a1):
  return np.matmul(np.matmul(np.matmul(T_z(t2),T_y(t1)),T_x(t3,a0)),np.matmul(np.matmul(T_z(t4),T_y(t5)),T_x(0,a1)))
#  V = np.matrix([[1],[1],[1],[1]])
#  return np.matmul(T,V)

def test_1():
  old = MCP_rel_wrist(60,30,25,3)
  new = MCP(60,30,25,3)
  print("old:")
  for row in old.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))
  print()
  print("new:")
  for row in new.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))
  print()

def call_MCP_rel_wrist():
  MCP_rel_wrist(60,30,25,3)

def call_MCP():
  MCP(60,30,25,3)

def test_2():
  exec_time_old = timeit.timeit(call_MCP_rel_wrist, number=1)
  exec_time_new = timeit.timeit(call_MCP, number=1)
  print("exec_time_old (matmul):  {}".format(exec_time_old))
  print("exec_time_new (precomp): {}".format(exec_time_new))

#test_1()
#test_2()

def print_matrix(m):
  for row in m.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))

@jit
def hashable_matrix(m):
  l = []
  for row in m.A:
    l.append(row.tolist())
  
  return tuple(tuple(sublist) for sublist in l)

def test_3():
  old = PIP_rel_wrist(60,30,25,20,80,3,1)
  new = PIP(60,30,25,20,80,3,1)
  print("old:")
  for row in old.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))
  print()
  print("new:")
  for row in new.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))
  print()

def call_PIP_rel_wrist():
  PIP_rel_wrist(60,30,25,20,80,3,1)

def call_PIP():
  PIP(60,30,25,20,80,3,1)

def test_4():
  exec_time_old = timeit.timeit(call_PIP_rel_wrist, number=1)
  exec_time_new = timeit.timeit(call_PIP, number=1)

  print("exec_time_old (matmul):  {}".format(exec_time_old))
  print("exec_time_new (precomp): {}".format(exec_time_new))

#test_3()
#test_4()


"""
temp = {}
for i in tqdm(range(total)):
  key = i%100
  if key in temp:
    temp[key].append(i)
  else:
    temp[key] = [i]  
"""
@jit
def comb(t1,t2,t3,t4,t5):
  for r in product(t1,t2,t3,t4,t5):
      yield r

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

"""
def compute(t1_r, t2_r, t3_r, t4_r, t5_r, a0, a1):
  out = [None]*12382518786
  for t1 in t1_r:
      for t2 in t2_r:
          for t3 in t3_r:
              for t4 in t4_r:
                  for t5 in tqdm(t5_r):
                    t1,t2,t3,t4,t5 = next(all_t)
                    t6 = t3
                    pip = PIP(t1,t2,t3,t4,t5,a0,a1)

                    loc_MCP_rel_wrist = MCP(t1,t2,t3,a0).A[:-1,-1].tolist()
                    loc_PIP_rel_wrist = pip.A[:-1,-1].tolist()

                    key = hashable_matrix(pip[:-1,:3])

                    out[i] = (key,loc_MCP_rel_wrist,loc_PIP_rel_wrist)
  return out
"""

def create_pcloud():
  for t1,t2,t3,t4,t5 in tqdm(product(t1_r, t2_r, t3_r, t4_r, t5_r)):
    t6 = t3
    pip = PIP(t1,t2,t3,t4,t5,a0,a1)

    loc_MCP_rel_wrist = MCP(t1,t2,t3,a0).A[:-1,-1].tolist()
    loc_PIP_rel_wrist = pip.A[:-1,-1].tolist()
 
    # Prints rotation matrix and locations 
    if t1 == -60 and t2 == -30:
      print("MCP:")
      print_matrix(MCP(t1,t2,t3,a0))
      print()
      print("PIP:")
      print_matrix(PIP(t1,t2,t3,t4,t5,a0,a1))
      print() 

      print('({},{},{},{}) -> {}'.format(t1,t2,t3,a0,loc_MCP_rel_wrist))
      print('({},{},{},{},{},{},{}) -> {}'.format(t1,t2,t3,t4,t5,a0,a1,loc_PIP_rel_wrist))

    key = hashable_matrix(pip[:-1,:3])
    if key in MCP_point_cloud:
      print("key exists")
      MCP_point_cloud[key].append(loc_MCP_rel_wrist)
    else:
      MCP_point_cloud[key] = [loc_MCP_rel_wrist]

    if key in PIP_point_cloud:
      PIP_point_cloud[key].append(loc_PIP_rel_wrist) 
    else:
      PIP_point_cloud[key] = [loc_PIP_rel_wrist]
    count += 1

"""
for t1,t2,t3,t4,t5 in tqdm(product(t1_r, t2_r, t3_r, t4_r, t5_r)):
  t6 = t3
  pip = PIP(t1,t2,t3,t4,t5,a0,a1)

  loc_MCP_rel_wrist = MCP(t1,t2,t3,a0).A[:-1,-1].tolist()
  loc_PIP_rel_wrist = pip.A[:-1,-1].tolist()
 
  Prints rotation matrix and locations 
  if t1 == -60 and t2 == -30:
    print("MCP:")
    print_matrix(MCP(t1,t2,t3,a0))
    print()
    print("PIP:")
    print_matrix(PIP(t1,t2,t3,t4,t5,a0,a1))
    print() 

    print('({},{},{},{}) -> {}'.format(t1,t2,t3,a0,loc_MCP_rel_wrist))
    print('({},{},{},{},{},{},{}) -> {}'.format(t1,t2,t3,t4,t5,a0,a1,loc_PIP_rel_wrist))

  key = hashable_matrix(pip[:-1,:3])
  if key in MCP_point_cloud:
    print("key exists")
    MCP_point_cloud[key].append(loc_MCP_rel_wrist)
  else:
    MCP_point_cloud[key] = [loc_MCP_rel_wrist]

  if key in PIP_point_cloud:
    PIP_point_cloud[key].append(loc_PIP_rel_wrist) 
  else:
    PIP_point_cloud[key] = [loc_PIP_rel_wrist]
  count += 1

# Point Cloud Mapping
for t3 in tqdm(np.arange(0,181,1), leave=False):
  if count == stop: break
  for t4 in tqdm(np.arange(-20,26,1), leave=False):
    if count == stop: break
    for t5 in tqdm(np.arange(-120,121,1), leave=False):
      if count == stop: break
      for t1 in tqdm(np.arange(-60,61,1), leave=False):
        for t2 in tqdm(np.arange(-30,21,1), leave=False):    
          t6 = t3

          loc_MCP_rel_wrist = MCP(t1,t2,t3,a0).A[:-1,-1].tolist()
          loc_PIP_rel_wrist = PIP(t1,t2,t3,t4,t5,a0,a1).A[:-1,-1].tolist()
         
          Prints rotation matrix and locations 
          if t1 == -60 and t2 == -30:
            print("MCP:")
            print_matrix(MCP(t1,t2,t3,a0))
            print()
            print("PIP:")
            print_matrix(PIP(t1,t2,t3,t4,t5,a0,a1))
            print() 

            print('({},{},{},{}) -> {}'.format(t1,t2,t3,a0,loc_MCP_rel_wrist))
            print('({},{},{},{},{},{},{}) -> {}'.format(t1,t2,t3,t4,t5,a0,a1,loc_PIP_rel_wrist))
         

          key = hashable_matrix(PIP(t1,t2,t3,t4,t5,0,0))
          if key in MCP_point_cloud:
            MCP_point_cloud[key].append(loc_MCP_rel_wrist)
          else:
            MCP_point_cloud[key] = []
            MCP_point_cloud[key].append(loc_MCP_rel_wrist)
  
          if key in PIP_point_cloud:
            PIP_point_cloud[key].append(loc_PIP_rel_wrist) 
          else:
            PIP_point_cloud[key] = []
            PIP_point_cloud[key].append(loc_PIP_rel_wrist)
      count += 1

def test_1():
  old = MCP_rel_wrist(60,30,25,3)
  new = MCP(60,30,25,3)
  print("old:")
  for row in old.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))
  print()
  print("new:")
  for row in new.A:
    print("| {:<10f} | {:<10f} | {:<10f} | {:<10f} |".format(row[0],row[1],row[2],row[3]))
  print()

def call_MCP_rel_wrist():
  MCP_rel_wrist(60,30,25,3)

def call_MCP():
  MCP(60,30,25,3)

def test_2():
  exec_time_old = timeit.timeit(call_MCP_rel_wrist, number=1)
  exec_time_new = timeit.timeit(call_MCP, number=1)

  print("exec_time_old (matmul):  {}".format(exec_time_old))
  print("exec_time_new (precomp): {}".format(exec_time_new))

test_1()
test_2()
"""

if __name__ == "__main__":
  MCP_point_cloud = {}
  PIP_point_cloud = {}
  a0 = 3
  a1 = 1

  count = 0
  stop = 10000000

  t1_r = cupy.arange(-60,61,1)
  t2_r = cupy.arange(-30,21,1)
  t3_r = cupy.arange(0,181,1)
  t4_r = cupy.arange(-20,26,1)
  t5_r = cupy.arange(-120,121,1)

  total = len(t1_r)*len(t2_r)*len(t3_r)*len(t4_r)*len(t5_r)
  print("Total combinations: {}".format(total))

  num_cores = multiprocessing.cpu_count()
  print(num_cores)
  mempool = cupy.get_default_memory_pool()
  print(mempool.used_bytes())              # 0
  print(mempool.total_bytes())  
  all_t_gpu = cupy.array(cupy.meshgrid(t1_r, t2_r, t3_r, t4_r, t5_r)).T.reshape(-1,5)

  print(all_t_gpu[:5,:])
  print("Finished creating the grid")
  
  splits = 1000
  step = len(all_t_gpu) / splits
  for i in range(splits):
    results = compute(cupy.asnumpy(all_t_gpu[i*step:(i+1)*step,:]), a0, a1)
  compute.parallel_diagnostics(level=4)
  print("Finished computing")
  #all_t = comb(t1_r, t2_r, t3_r, t4_r, t5_r)
  #create_pcloud()
  #result = compute(t1_r, t2_r, t3_r, t4_r, t5_r, a0, a1)
#  def compute_wrapper(args):
#    return compute(*args)
#
#  pool = multiprocessing.Pool(processes=7)
#  multiprocessing.util.log_to_stderr(10)
#  manager = multiprocessing.Manager()
#
#  L= manager.list()
#
#  [pool.apply_async(compute, args=[t1,t2,t3,t4,t5,a0,a1,L]) for t1,t2,t3,t4,t5 in tqdm(all_t)]
#  pool.close()
#  pool.join()
  #processed_list = Parallel(n_jobs=num_cores)(delayed(compute)(t1,t2,t3,t4,t5,a0,a1) for t1,t2,t3,t4,t5 in tqdm(all_t))
  #data = []
  #with multiprocessing.Pool(7) as p:
  #  data = p.imap_unordered(compute_wrapper, all_t)
  #
  #for element in tqdm(data):
  #  pass

  """Plot MCP and PIP for one orientation """
  def components(v):
    x = list(map(lambda item: item[0],v))
    y = list(map(lambda item: item[1],v))
    z = list(map(lambda item: item[2],v))
    return x, y, z
  """  
  MCP_list = list(MCP_point_cloud.items())

  random_MCP = random.choice(MCP_list)
  random_PIP = PIP_point_cloud[random_MCP[0]]

  MCP_x, MCP_y, MCP_z = components(random_MCP[1])
  PIP_x, PIP_y, PIP_z = components(random_PIP)

  fig = plt.figure()
  ax = fig.add_subplot(111, projection='3d')

  ax.scatter(MCP_x,MCP_y,MCP_z,c='r',marker='o',label='MCP')
  ax.scatter(PIP_x,PIP_y,PIP_z,c='b',marker='o',label='PIP')

  ax.set_title('(t4,t5,t6): {}'.format(random_MCP[0]))
  ax.set_xlabel('X Label')
  ax.set_ylabel('Y Label')
  ax.set_zlabel('Z Label')

  plt.show()
  """
