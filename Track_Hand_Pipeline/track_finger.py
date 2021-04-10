import sys
import os
import MPHandsTracker as mpht
import argparse
import numpy as np
import multiprocessing
import yaml
import logging

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../Video_Datasets/ASLLVD/')
import common

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../logging/')
import configure as logs

from joblib import Parallel, delayed

# Configure Logger
logs.configure()

"""
Should be used to build and save finger dataset

Parameters
-----------
video   : str, optional
    video file used for finger tracking (default: http://vlm1.uta.edu/~haijing/asl/camera1/mov/ASL_2008_05_12a/scene4-camera1.mov)
s       : [Boolean Flag], optional
    indicates that the dataset should be saved
verbose : [Boolean Flag], optional
    indicates that more fine grain information should be printed
name    : str
    name used when saving the dataset
"""

ds = []

def save(name, header='x,y,z,d2x,d2y,d2z'):
  """
  Saves the data in global variable `ds` into a csv file

  Parameters
  ----------
  name   : str
      name of file to be saved within a datasets folder
  header : str, optional
      header for the csv file (default: 'x,y,z,dx,dy,dz,d2x,d2y,d2z')
  """ 
  global ds
  
  dataset = np.asarray(ds, dtype=np.float32)
  path = "../datasets/{}.csv".format(name)
  h = header if not os.path.isfile(path) else ''
  with open(path,'a') as f:
    np.savetxt(f, dataset, delimiter=",",header=h,comments='')


def collect(finger_data, accel_data):
  """
  Stores position, velocity, and acceleration data to
   global variable ds

  Parameters
  ----------
  finger_data : list
      list containing 3D positional data (x,y,z)
  accel_data  : list
      list containing 3D acceleration data (d2x,d2y,d2z)
  """
  global ds
 
  res = [] 
  for i in range(len(finger_data)):
    if i > 2:
      temp = []
      temp.extend(finger_data[i])
      temp.extend(accel_data[i-3])
      ds.append(temp)
      res.append(temp)
  return res

def track(video, verbose=False, should_save=False, video_count=1, video_total_count=1):
  """
  Runs MediaPipe Hands tracker on video

  Parameters
  ----------
  video       : str
      file path to video that will be used for tracking
  verbose     : bool, optional
      if true, position, velocity, and acceleration data for each
      frame will be printed to the console (default: False)
  should_save : bool, optional
      if true, position, velocity, and acceleration data for each
      frame will be collected into global variable ds (default: False)
  """
  logger = logging.getLogger("basic") 
  tracker = mpht.MPHandsTracker(verbose)
  logger.info("Processing video {} / {}".format(video_count, video_total_count))
  try:
    tracker.run(video)
    finger_data, accel_data = tracker.logRelativeAcceleration(handedness="Right")
   
    assert len(finger_data)-2 == len(accel_data), "Invalid finger data/acceleration data lengths ({}/{})".format(len(finger_data),len(accel_data)) 
  
    if verbose:
      for i in range(len(finger_data)):
        logger.debug("hand_data   -> x: {:.6f}, y: {:.6f}, z: {:.6f}".format(finger_data[i][0], finger_data[i][1], finger_data[i][2]))
        if i > 2:
          logger.debug("accel_data  -> x: {:.6f}, y: {:.6f}, z: {:.6f}".format(accel_data[i-3][0], accel_data[i-3][1], accel_data[i-3][2]))

    if should_save:
      res = collect(finger_data,accel_data)
      return res
  except ValueError as e:
    logger.error("ValueError: {}".format(e), exc_info=True)
    sys.exit()
  except Exception as e:
    logger.error("Exception: {}".format(e), exc_info=True)
    sys.exit()
  except:
    logger.error("Unexpected error:", sys.exc_info()[0])
    sys.exit() 

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='finger_tracking')
  parser.add_argument('--video', type=str, default="http://vlm1.uta.edu/~haijing/asl/camera1/mov/ASL_2008_05_12a/scene4-camera1.mov")
  parser.add_argument('--s', help='save acceleration data', action='store_true')
  parser.add_argument('--verbose', help='print extra information', action='store_true')
  parser.add_argument('--name', type=str, help='name for the dataset to be saved')
  parser.add_argument('--allASLLVD', help='run finger tracker on all ASLLVD videos', action='store_true')
  parser.add_argument('--IPN', help='run finger tracker on IPN videos', action='store_true')
  parser.add_argument('--IPN_start', type=int, default=0, help='index of video to start at (Starting at 1')
  parser.add_argument('--IPN_end', type=int, help='index of video to end at (Starting at 1')
  parser.add_argument('--ASLLVD_start', type=int, default=1, help='index of video to start at (Starting at 1')
  parser.add_argument('--ASLLVD_end', type=int, default=common.sizeOfASLLVD(), help='index of video to end at (Starting at 1')
  parser.add_argument('--parallelize', help='run parallelized finger for faster processing', action='store_true')

  args = parser.parse_args()

  if args.verbose:
    logger = logging.getLogger("debug")
  else:
    logger = logging.getLogger("basic")    

  video_count = 1
  video_total_count = common.sizeOfASLLVD()

  num_cores = multiprocessing.cpu_count()

  # TODO: For parallelized versions, should check to make sure ds is built correctly
  #        because collect(...) still appends to ds
  if args.allASLLVD:
    for i, session in enumerate(common.SESSIONS):
      if video_count >= args.ASLLVD_end:
        logger.debug('{} videos completed'.format(video_count))
        break
      for camera in common.CAMERAS[i]:
        if int(camera) != 1: # Other cameras are at different angles. 
          break
        if video_count >= args.ASLLVD_end:
          break
        if video_count < args.ASLLVD_start:
          video_count += len(common.SCENES[i])
          continue
        inputs = common.SCENES[i]

        if args.parallelize:
          processed_list = Parallel(n_jobs=num_cores)(delayed(track)(common.ASLLVD(session,inputs[j],camera),args.verbose,args.s,video_count+j,video_total_count) for j in range(len(inputs)))
          ds.extend([item for sublist in processed_list for item in sublist if len(item) > 0])
          video_count += len(common.SCENES[i])
        else:
          for scene in inputs:
            if video_count < args.ASLLVD_start:
              continue
            if video_count-1 >= args.ASLLVD_end:
              break
            video = common.ASLLVD(session,scene,camera)
            track(video,args.verbose,args.s,video_count,video_total_count)
            video_count += 1
  elif args.IPN:
    logger.debug("Training on IPN video dataset")
    directory = '../Video_Datasets/IPN/'
    paths = []
    for filename in os.listdir(directory):
      if filename.endswith(".avi"):
        paths.append(os.path.join(directory, filename))
      else:
        continue
    video_total_count = len(paths)
    if args.parallelize:
      inputs = []
      if args.IPN_end is not None:
        inputs = paths[args.IPN_start:args.IPN_end]
      else:
        inputs = paths[args.IPN_start:]
      processed_list = Parallel(n_jobs=num_cores)(delayed(track)(inputs[j],args.verbose,args.s,video_count+j,video_total_count) for j in range(len(inputs)))

      ds.extend([item for sublist in processed_list for item in sublist if len(item) > 0])
      video_count += len(inputs)
    else:
      start_i = 0
      if args.IPN_start is not None: start_i = args.IPN_start
      for path in paths[start_i:]:
        track(path, args.verbose, args.s, video_count, video_total_count)
        video_count += 1

  else:
    track(args.video,args.verbose,args.s)
    
  if args.s:
    save(args.name)

  
