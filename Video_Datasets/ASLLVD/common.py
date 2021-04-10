import itertools
from itertools import chain

"""
Common variables to be shared throughout the repository
"""

SESSIONS = ['ASL_2006_10_10','ASL_2007_05_24','ASL_2008_01_11','ASL_2008_01_18','ASL_2008_02_01','ASL_2008_02_15','ASL_2008_02_29','ASL_2008_03_28','ASL_2008_05_16','ASL_2008_05_12a','ASL_2008_05_12b','ASL_2008_05_29a','ASL_2008_05_29b','ASL_2008_06_10','ASL_2008_05_21','ASL_2008_08_04','ASL_2008_08_06','ASL_2008_08_13','ASL_2008_08_13_session2','ASL_2008_10_03']
SCENES = [range(2,9),range(6,13),range(1,86),range(1,52),range(1,59),range(1,53),range(1,56),list(chain(range(1,3),range(10,50))),range(1,14),range(1,52),range(1,21),range(1,53),range(1,22),range(1,25),range(1,15),range(1,54),range(1,52),range(1,38),range(1,15),range(1,17)]
CAMERAS = [range(1,4),range(1,4),range(1,4),range(1,4),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2),range(1,2)]

def sizeOfASLLVD():
  count = 0
  for i in range(len(SESSIONS)):
    for camera in CAMERAS[i]:
      for scene in SCENES[i]:
        count += 1
  return count

def ASLLVD(session,scene,camera):
  """
  returns filepath to video specified by the session, scene, and camera
  
  Parameters
  ----------
  session : str
      session name
  scene   : int
      scene number
  camera  : int
      camera number

  Returns
  -------
  str
      a string denoting the filepath to the video within the American 
      Sign Language Lexicon Video Dataset corresponding to the given 
      session, scene, and camera
  """
  return "http://vlm1.uta.edu/~haijing/asl/camera1/mov/{}/scene{}-camera{}.mov".format(session,scene,camera)
