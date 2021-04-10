# Synth3DFingerTracking
IMU-based 3D finger tracker trained on synthetic IMU dataset created from Monocular 3D finger tracking

## Tracking Pipeline
**track_finger.py** creates the synthentic IMU dataset. Check the script's arguments by running:<br />

`python track_finger.py --help`

## Training Pipeline 
**train_finger_rnn.py** trains a simple RNN from the provided synthetic IMU data. Check the script's arguments by running:<br />

`python train_finger_rnn.py --help`
