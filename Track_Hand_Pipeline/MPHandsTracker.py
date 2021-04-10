import sys
import math
import time
import logging
import cv2
import argparse
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt #visualisation
import pandas as pd
import matplotlib.gridspec as gridspec

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../logging/')
import configure as logs

from scipy.signal import butter,filtfilt
from progress.bar import Bar

# Configure logger
logs.configure()

""" Used to build synthetic training set """

HandLandmarkLabel = [
"WRIST",
"THUMB_CMC",
"THUMB_MCP",
"THUMB_IP",
"THUMB_TIP",
"INDEX_FINGER_MCP",
"INDEX_FINGER_PIP",
"INDEX_FINGER_DIP",
"INDEX_FINGER_TIP",
"MIDDLE_FINGER_MCP",
"MIDDLE_FINGER_PIP",
"MIDDLE_FINGER_DIP",
"MIDDLE_FINGER_TIP",
"RING_FINGER_MCP",
"RING_FINGER_PIP",
"RING_FINGER_DIP",
"RING_FINGER_TIP",
"PINKY_MCP",
"PINKY_PIP",
"PINKY_DIP",
"PINKY_TIP"
]

mp_drawing = mp.solutions.drawing_utils
mp_hands = mp.solutions.hands

def butter_lowpass_filter(data, cutoff, fs, order):
  nyq = 0.5 * fs  # Nyquist Frequency
  normal_cutoff = cutoff / nyq
  # Get the filter coefficients 
  b, a = butter(order, normal_cutoff, btype='low', analog=False)
  y = filtfilt(b, a, data)
  return y

def cfd_kernel(input_sequence, i, h, frames, col):
  numerator = (input_sequence[i-1,col] + input_sequence[i+1,col] - 2*input_sequence[i,col])
  denominator = (h*float(frames[i+1] - frames[i-1]))**2
  assert denominator != 0, "Divide by zero. frame_i+1 = {}, frame_i-1 = {} (i)".format(frames[i+1],frames[i-1],i)
  return numerator / denominator

def cfd_2(input_sequence, h, frames):
  """ Second-Order Central Finite Difference """
  input_seq_len = len(input_sequence)
  if input_seq_len < 3: 
    raise Exception("Input Sequence was too short: len = {}".format(input_seq_len))
 
  iter_range = range(1,input_seq_len-1) # Range is from the second element to the second to last element

  finite_diff_iter_1 = (cfd_kernel(input_sequence, i, h, frames, 0)  for i in iter_range)
  finite_diff_iter_2 = (cfd_kernel(input_sequence, i, h, frames, 1)  for i in iter_range)
  finite_diff_iter_3 = (cfd_kernel(input_sequence, i, h, frames, 2)  for i in iter_range)
  
  d2x = np.fromiter(finite_diff_iter_1,np.float64)
  d2y = np.fromiter(finite_diff_iter_2,np.float64)
  d2z = np.fromiter(finite_diff_iter_3,np.float64)

  return np.stack((d2x,d2y,d2z), axis=-1)


class MPHandsTracker:
  
  def __init__(self, verbose=False, show_plots=False):
    self.name = "MPHandsTracker"
    self.raw_output_data = []
    self.output_data = []
    self.fps = 0.0
    self.spf = 0.0
    # Now describes the number of frames that the left or right hand DO show up in
    self.droppedFrames = (0,0) # (left, right)
    self.numFrames = 0

    self.show_plots = show_plots

    self.ptcm = 0.0 # Pixel to cm conversation factor
    self.img_width = 0.0
    self.img_height = 0.0

    if verbose:
      self.logger = logging.getLogger("debug")
      self.logger.debug("Logging set to DEBUG")
      self.logger.info("INFO")
    else:
      self.logger = logging.getLogger("basic")

  def getOutput(self):
    return self.output_data
  
  def getDroppedFrames(self):
    """ Return the number of frames with no left hand and no right hand (separately) """
    return (self.numFrames - self.droppedFrames[0], self.numFrames - self.droppedFrames[1])

  def getTotalFrames(self):
    """ Get total frames of video """
    return self.numFrames
 
  def best_hand_landmark(self, multi_hand_landmarks, multi_handedness, handedness):
    best = None
    max_score = 0
    for index, hand_landmarks in enumerate(multi_hand_landmarks):
      given_handedness = str(multi_handedness[index]).split('\n')[3].strip()
      if handedness in given_handedness:
        score = float(str(multi_handedness[index]).split('\n')[2].strip().split("score:")[1].strip())
        if score > max_score:
          max_score = score
          best = hand_landmarks

    return best
 
  def calculate_pixel_to_cm(self, results, handedness):
    """ Calculates conversion factor from pixels to centimeters

    This is used to translation finger joint positions relative to wrist
     from pixels to approximate cm
    """
    # 1. Get the wrist and index_mcp corresponding with the highest hand score for the given handedness
    best_landmark = self.best_hand_landmark(results.multi_hand_landmarks, results.multi_handedness, handedness)
    if best_landmark is None: return 0 # Pixel conversion factor could never be 0, so this should be okay as a failure value   

    wrist = best_landmark.landmark[0]
    index_mcp = best_landmark.landmark[5]

    # 2. Calculate the conversion factor
    dx = abs(index_mcp.x*self.img_width - wrist.x*self.img_width)
    dy = abs(index_mcp.y*self.img_height - wrist.y*self.img_height)  
    dz = abs(index_mcp.z*self.img_width)
    pixel_length = math.sqrt(dx**2 + dy**2 + dz**2)
    self.logger.debug("Pixel length between Index Finger MCP and wrist: {}".format(pixel_length))
    actual_length = 11.0  # Assume standard length between index finger mcp and wrist to be 11 cm
    
    assert pixel_length != 0, "wrist and index_mcp have same coordinate"

    # 3. Return conversion factor
    return actual_length / pixel_length
    
  def run(self, file_list):
    """
    Runs the MediaPipe Hands Tracker on static images
    """

    self.output_data = []
    hands = mp_hands.Hands(
        static_image_mode=True,
        max_num_hands=2,
        min_detection_confidence=0.5)
    for idx, file in enumerate(file_list):
      # Read an image, flip it around y-axis for correct handedness output (see
      # above).
      image = cv2.flip(cv2.imread(file), 1)
      # Convert the BGR image to RGB before processing.
      results = hands.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
      self.results.append(results)

      # Print handedness and draw hand landmarks on the image.
      self.logger.debug("Handedness: {}".format(results.multi_handedness))
      if not results.multi_hand_landmarks:
        continue
      image_hight, image_width, _ = image.shape
      annotated_image = image.copy()
      for hand_landmarks in results.multi_hand_landmarks:
        self.logger.debug("hand_landarks: {}".format(hand_landmarks))
        self.logger.debug("Index finger tip coordinates: ({},{},{})".format(hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].x * image_width,
                                                                            hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP].y * image_hight))
        mp_drawing.draw_landmarks(
            annotated_image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      cv2.imwrite(
          '/tmp/annotated_image' + str(idx) + '.png', cv2.flip(annotated_image, 1))
    hands.close()

  def run(self, file="", progress_title="Processing video", mirrored=True):
    """
    Runs the MediaPipe Hands Tracker on a file or webcam input if no file is provided
    """

    self.output_data = []   # Reset output data each time run is called
    word = file.split('.')[0]   # Extracts the word in sign language from the filename
    hands = mp_hands.Hands(
        min_detection_confidence=0.75, min_tracking_confidence=0.75)
    cap = cv2.VideoCapture(file if len(file) != 0 else 0)   # If no file is provided, the webcam will be used
    self.fps = cap.get(cv2.CAP_PROP_FPS)
    self.spf = 1.0 / self.fps
    self.logger.debug("fps: {}".format(self.fps))
    

    self.numFrames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    ideal_frame = 100   # Ideal frame for pixel to cm conversion factor
    frame_count = 0 
    while cap.isOpened():
      if self.fps == 0:
        raise Exception("ERROR - fps is 0. The video probably could not be read")
      success, image = cap.read()
      if not success:
        self.logger.info("Ignoring empty camera frame.")
        # If loading a video, use 'break' instead of 'continue'.
        break
      
        
      # Flip the image horizontally for a later selfie-view display, and convert
      # the BGR image to RGB.
      image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
      self.img_height, self.img_width, _ = image.shape
      self.logger.debug("cam image={}x{}".format(image.shape[1],image.shape[0]))
      # To improve performance, optionally mark the image as not writeable to
      # pass by reference.
      image.flags.writeable = False
      results = hands.process(image)
      
      if results.multi_handedness != None:
        if len(self.raw_output_data) != 0:
          frame_repeat_error = "Current frame count is not greater than a previous frame's frame count. (fc: {}, prev_fc:{})".format(frame_count,self.raw_output_data[-1][2])
          assert frame_count > self.raw_output_data[-1][2], frame_repeat_error
        
        self.raw_output_data.append((results.multi_handedness,results.multi_hand_landmarks,frame_count))
        
 
        if frame_count == ideal_frame:      # Ideal frame for pixel to cm conversion factor
          # results, handedness
          self.ptcm = self.calculate_pixel_to_cm(results, "Right")
          if self.ptcm == 0: ideal_frame += 1  # This will cause the pixel to cm conversion factor to be recomputed the next frame (or until
                                               #  a valid frame exists

  
        if len(results.multi_handedness) > 1:
          handedness1 =  str(results.multi_handedness[0]).split('\n')[3].strip()
          handedness2 =  str(results.multi_handedness[1]).split('\n')[3].strip()
          if handedness1 == handedness2:
            self.logger.debug("{} hand predicted twice. (frame #: {})".format(handedness1, frame_count))
            score1 = str(results.multi_handedness[0]).split('\n')[2].strip().split("score:")[1].strip()
            score2 = str(results.multi_handedness[1]).split('\n')[2].strip().split("score:")[1].strip()
            self.logger.debug("hand1 score: {}\thand2 score: {}".format(score1,score2))
        
        prev_droppedFrames = self.droppedFrames
        for handedness in results.multi_handedness:
          given_handedness = str(handedness).split('\n')[3].strip()
          if "Left" in given_handedness:
            if mirrored: # Correct handedness
              self.droppedFrames = (prev_droppedFrames[0]+1,prev_droppedFrames[1])
            else: # Opposite handedness
              self.droppedFrames = (prev_droppedFrames[0],prev_droppedFrames[1]+1)
          elif "Right" in given_handedness:
            if mirrored: # Correct handedness
              self.droppedFrames = (prev_droppedFrames[0],prev_droppedFrames[1]+1)
            else: # Opposite handedness
              self.droppedFrames = (prev_droppedFrames[0]+1,prev_droppedFrames[1]) 
      
      frame_count += 1
        
      key = cv2.waitKey(1)
      if key & 255 == ord('q'):
        break
      
      # Draw the hand annotations on the image.
      image.flags.writeable = True
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      image_hight, image_width, _ = image.shape
      if results.multi_hand_landmarks:
        self.logHands(results, image)
        for hand_landmarks in results.multi_hand_landmarks:
          mp_drawing.draw_landmarks(
              image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
      
      if self.show_plots:
        cv2.imshow("MediaPipe Hands - {}".format(word), image)
      
      if cv2.waitKey(5) & 0xFF == 27:
        break
    
      
    hands.close()
    cap.release()

  def logRelativeAcceleration(self, joint="INDEX_FINGER_TIP", handedness="Right"):
    frames = []
    X = np.empty((0,3), np.float64)
    d2X = np.empty((0,3), np.float64)

    # 1. Get index for joint 
    joint_index = 0
    for j in mp_hands.HandLandmark:
      if HandLandmarkLabel[j] == joint:
        joint_index = j

    # 2. Collect frame data and positional data for given joint of given hand with max score
    for frame in self.raw_output_data:
      if len(frame[1]) > 2: self.logger.warning("More than 2 hands were detected.")
      
      best_landmark = self.best_hand_landmark(frame[1], frame[0], handedness)
      if best_landmark is None: continue
      wrist = best_landmark.landmark[0]

      frames.append(frame[2])
      dx = (best_landmark.landmark[joint_index].x - wrist.x) * self.img_width
      dy = (best_landmark.landmark[joint_index].y - wrist.y) * self.img_height
      dz = best_landmark.landmark[joint_index].z * self.img_width
      pixel_length = math.sqrt(dx**2 + dy**2 + dz**2)
      X = np.append(X, [[(best_landmark.landmark[joint_index].x - wrist.x) * self.img_width * self.ptcm,
                         (best_landmark.landmark[joint_index].y - wrist.y) * self.img_height * self.ptcm,
                         best_landmark.landmark[joint_index].z * self.img_width * self.ptcm]], axis=0)    # z-coord is already relative to wrist z-coord

    if X.shape[0] == 0: self.logger.warning("joint position data for {} {} is empty".format(handedness,joint))
    self.logger.debug("Joint positions: {}".format(X[:5,:]))

    # 3. Apply butterworth low-pass filter over positional data to correct for noise
    cutoff = 0.1
    fs = 10
    order = 2
    x = butter_lowpass_filter(X[:,0],0.1,10,2)
    y = butter_lowpass_filter(X[:,1],0.1,10,2)
    z = butter_lowpass_filter(X[:,2],0.1,10,2)
    X_filt = np.stack((x,y,z), axis=-1)

    self.logger.debug("Filtered Joint positions: {}".format(X_filt[:5,:]))

    if self.show_plots:
      self.create_plots(X,X_filt)
      plt.show()

    # 4. Calculate acceleration through second order central finite difference
    try:
      d2X = cfd_2(X_filt, self.spf, frames)
      self.logger.debug("Acceleration data: {}".format(d2X[:5,:]))

      # 5. Apply another butterworth low-pass filter over acceleration data to correct for noise
      cutoff = 0.5
      fs = 10
      order = 2
      d2x = butter_lowpass_filter(d2X[:,0],0.5,10,2)
      d2y = butter_lowpass_filter(d2X[:,1],0.5,10,2)
      d2z = butter_lowpass_filter(d2X[:,2],0.5,10,2)

      d2X_filt = np.stack((d2x,d2y,d2z), axis=-1)
      self.logger.debug("Filtered Accleration data: {}".format(d2X_filt[:5,:]))
 
      if self.show_plots:
        self.create_plots(d2X, d2X_filt)
        plt.show()

      return (X_filt,d2X_filt)
    except Exception as e:
      self.logger.error("Acceleration could not be calculated from position data: {}".format(e))
      raise Exception("Acceleration could not be calculated from position data.")
  
  
  def logHands(self, results, image):
    image_hight, image_width, _ = image.shape
    frame_info = []
    for index, hand_landmarks in enumerate(results.multi_hand_landmarks):
      handedness = str(results.multi_handedness[index]).split('\n')
      self.logger.debug(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
      self.logger.debug("")
      self.logger.debug(handedness[3].strip())
      frame_info.append("0" if "Left" in handedness[3] else "1") 
      self.logger.debug(handedness[2].strip())
      self.logger.debug("")
      for joint in mp_hands.HandLandmark:
        self.logger.debug('%s: (%.2f, %.2f, %.2f)', 
                    HandLandmarkLabel[joint], 
                    hand_landmarks.landmark[joint].x * image_width, 
                    hand_landmarks.landmark[joint].y * image_hight, 
                    hand_landmarks.landmark[joint].z * image_width)
        frame_info.append(str(hand_landmarks.landmark[joint].x))
        frame_info.append(str(hand_landmarks.landmark[joint].y))
        frame_info.append(str(hand_landmarks.landmark[joint].z))
      self.logger.debug("")
      self.logger.debug("<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<")
    filler_data = []
    if len(frame_info) < 128:
      if frame_info[0] == "0":
        filler_data.append("1")
        filler_data.extend(["0" for i in range(63)])
        frame_info.extend(filler_data)
      else:
        filler_data = ["0" for i in range(64)]
        filler_data.extend(frame_info)
        frame_info = filler_data
    else:
      if frame_info[0] == "1":
        filler_data = frame_info[:64]
        frame_info = frame_info[64:]
        frame_info.extend(filler_data)
    if len(frame_info) == 128:
      self.output_data.append(",".join(frame_info) + '\n')


  def create_plots(self,X1,X2):
    fig = plt.figure(constrained_layout=True, figsize=(18,5))
    axs = fig.subplots(1,3)
    fig.add_gridspec(nrows=1,ncols=3,left=0.125,bottom=0.11,right=0.9,top=0.88,wspace=0.3,hspace=0.2)

    df1 = pd.DataFrame({"x": X1[:,0], "y": X1[:,1], "z": X1[:,2]}).reset_index()
    df2 = pd.DataFrame({"x": X2[:,0], "y": X2[:,1], "z": X2[:,2]}).reset_index()

    axs[0].plot(df1.index, df1["x"], label='x')
    axs[0].plot(df2.index, df2["x"], label='filtered')
    axs[0].set_xlabel('frame number')
    axs[0].set_ylabel('x coordinate relative to wrist')

    axs[1].plot(df1.index, df1["y"], label='y')
    axs[1].plot(df2.index, df2["y"], label='filtered')
    axs[1].set_xlabel('frame number')
    axs[1].set_ylabel('y coordinate relative to wrist')

    axs[2].plot(df1.index, df1["z"], label='z')
    axs[2].plot(df2.index, df2["z"], label='filtered')
    axs[2].set_xlabel('frame number')
    axs[2].set_ylabel('z coordinate relative to wrist')
  
  """ Deprecated (as of 3/19/21) """    
  def logAcceleration(self, joint="INDEX_FINGER_TIP", handedness="Right"):
    X = np.empty((0,3), np.float64)
    dX = np.empty((0,3), np.float64)
    d2X = np.empty((0,3), np.float64)

    joint_index = 0
    for j in mp_hands.HandLandmark:
      if HandLandmarkLabel[j] == joint:
        joint_index = j

    # Collect positional data
    for frame in self.raw_output_data:
      for index, hand_landmarks in enumerate(frame[1]):
        given_handedness = str(frame[0][index]).split('\n')[3].strip()
        if handedness in given_handedness:
          X = np.append(X, [[hand_landmarks.landmark[joint_index].x,hand_landmarks.landmark[joint_index].y,hand_landmarks.landmark[joint_index].z]], axis=0)

    # Calculate first derivative (velocity) assuming constant fps
    dX = linear_derivative(X, self.fps)

    # Calculate second derivative (acceleration) assuming constant fps
    d2X = linear_derivative(dX, self.fps)
    
    return (X, dX, d2X) 
  
  """ Deprecated (as of 3/11/21) """
  def logRelativeAcceleration_depr(self, joint="INDEX_FINGER_TIP", handedness="Right"):
    X = np.empty((0,3), np.float64)
    dX = np.empty((0,3), np.float64)
    d2X = np.empty((0,3), np.float64)

    joint_index = 0
    for j in mp_hands.HandLandmark:
      if HandLandmarkLabel[j] == joint:
        joint_index = j


    # Collect positional data
    for frame in self.raw_output_data:
      for index, hand_landmarks in enumerate(frame[1]):
        given_handedness = str(frame[0][index]).split('\n')[3].strip()
        wrist = hand_landmarks.landmark[0]
        if handedness in given_handedness:
          X = np.append(X, [[hand_landmarks.landmark[joint_index].x - wrist.x,
                             hand_landmarks.landmark[joint_index].y - wrist.y,
                             hand_landmarks.landmark[joint_index].z]], axis=0)    # z-coord is already relative to wrist z-coord

    if X.shape[0] == 0:
      print("len(X) is 0!!!")
    # Calculate first derivative (velocity) assuming constant fps
    dX = linear_derivative(X, self.fps)

    # Calculate second derivative (acceleration) assuming constant fps
    d2X = linear_derivative(dX, self.fps)

    return (X,dX,d2X)

""" Deprecated (as of 3/11/21) """
def linear_derivative(input_sequence, fps):
  X = np.array(input_sequence)
  dX = np.empty((0,3), np.float64)
  for i in range(0, len(X)-1, 1):
      dX_i = np.subtract(X[i+1],X[i])
      dX = np.append(dX, [dX_i], axis=0)
  spf = 1.0 / fps
  return np.divide(dX,spf)
