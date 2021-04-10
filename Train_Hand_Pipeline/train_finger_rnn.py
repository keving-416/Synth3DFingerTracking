import os
import sys
import logging
import argparse
import scipy
import pandas as pd
import numpy as np
import tensorflow as tf
import multiprocessing as mp
import matplotlib.pyplot as plt
import math

# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../logging/')
import configure as logs

from tqdm import tqdm
from tensorflow.keras import layers 
from tensorflow.keras.models import load_model 
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

# Configure Logger
logs.configure()

name = ""

gpus = tf.config.list_physical_devices('GPU')
if gpus:
  # Restrict TensorFlow to only use the first GPU
  try:
    tf.config.experimental.set_visible_devices(gpus[0], 'GPU')
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPU")
  except RuntimeError as e:
    # Visible devices must be set before GPUs have been initialized
    print(e)

def save_model(model):
  global name
  print(name)
  model.save("../models/{}.h5".format(name))

def load_model():
  model = load_model("model.h5")

def plot_predictions(predictions, y_test):
  x_pred = []
  y_pred = []
  z_pred = []
  x = []
  y = []
  z = []

  for prediction in predictions:
      x_pred.append(prediction[0])
      y_pred.append(prediction[1])
      z_pred.append(prediction[2])

  for coord in y_test:
      x.append(coord[0])
      y.append(coord[1])
      z.append(coord[2])

  plt.subplot(1, 3, 1)
  plt.plot(x, 'k-')
  plt.plot(x_pred, 'r-')
  plt.xlabel('Time [s]')
  plt.ylabel('x location [m]')
  plt.legend(['true x','predicted x'])

  plt.subplot(1, 3, 2)
  plt.plot(y, 'k-')
  plt.plot(y_pred, 'r-')
  plt.xlabel('Time [s]')
  plt.ylabel('y location [m]')
  plt.legend(['true y','predicted y'])
  
  plt.subplot(1, 3, 3)
  plt.plot(z, 'k-')
  plt.plot(z_pred, 'r-')
  plt.xlabel('Time [s]')
  plt.ylabel('z location [m]')
  plt.legend(['true z','predicted z'])

def plot_history(history):
    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model train vs validation loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'validation'], loc='upper right')

def train(seed=0, dataset='all_asllvd.csv', seq_hist_len=3, optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss=tf.keras.losses.LogCosh(), metrics=['accuracy'], batch_size=10, epochs=100):
    """ Simple RNN Model

    Parameters
    ----------
    seed : int
        The numpy randomizer seed (default is 0)
    dataset : str
        The name of the file containing the dataset (default is 'all_asllvd.csv')
    seq_hist_len : int
        The sequence history length for the LSTMs (default is 3)
    optimizer : tf.keras.optimizer
        The optimizer used for compiling the ML model (default is tf.keras.optimizers.Adam(learning_rate=0.0001))
    loss : tf.keras.losses
        The loss function to be used during training (default is tf.keras.losses.MeanSquaredError())
    metrics : [str]
        The list of metrics to be recorded during training (default is ['accuracy'])
    batch_size : int
        The batch size for training (default is 5)
    epochs : int
        The number of epochs for training (default is 100)
    """
    logger = logging.getLogger("basic")
    np.random.seed(seed)
   
    # 1. Load the dataset into a pandas dataframe 
    df = pd.read_csv("../datasets/{}".format(dataset))

    # 2. Split into training and test sets (80% train, 20% test)
    training_set, test_set = train_test_split(df, test_size=0.2)
    
    training_set['x'].plot.kde()
    training_set['y'].plot.kde()
    training_set['z'].plot.kde()

    #plt.show()

    # 3. Convert to numpy array 
    training_set = training_set.to_numpy()
    test_set = test_set.to_numpy()

    # 4. Normalize the data to between 0 and 1
    sc = MinMaxScaler(feature_range = (0, 1))
    training_set_scaled = sc.fit_transform(training_set)

    df_scaled = pd.DataFrame(data=training_set_scaled, columns=['x','y','z','d2x','d2y','d2z'])
    df_scaled['x'].plot.kde()
    df_scaled['y'].plot.kde()
    df_scaled['z'].plot.kde()
    #plt.show()

    df_scaled['d2x'].plot.kde()
    df_scaled['d2y'].plot.kde()
    df_scaled['d2z'].plot.kde()
    #plt.show()

    # 7. Collect training inputs and labels (75% of train)
    x_train = []
    y_train = []
    for i in tqdm(range(seq_hist_len,int(3*len(training_set_scaled)/4))):
        x_train.append(training_set_scaled[i-seq_hist_len:i,3:])
        y_train.append(training_set_scaled[i,:3])

    
    # 6. Collect validation inputs and labels (25% of train)
    x_val = []
    y_val = []
    for i in tqdm(range(int(3*len(training_set_scaled)/4),len(training_set_scaled))):
        x_val.append(training_set_scaled[i-seq_hist_len:i,3:])
        y_val.append(training_set_scaled[i,:3])

    # 7. Convert to numpy array
    x_train, y_train = np.asarray(x_train), np.asarray(y_train)
    x_val, y_val = np.asarray(x_val), np.asarray(y_val)
    x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 3))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], 3))
 
    # 8. Repeat previous steps with test set 
    sc_t = MinMaxScaler(feature_range = (0,1))
    test_set_scaled = sc_t.fit_transform(test_set)

    x_test = []
    y_test = []
    for i in tqdm(range(seq_hist_len,len(test_set_scaled))):
        x_test.append(test_set_scaled[i-seq_hist_len:i,3:])
        y_test.append(test_set_scaled[i,:3])

    x_test, y_test = np.asarray(x_test), np.asarray(y_test)
    x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 3))

    # 9. Create RNN Model
    model = tf.keras.Sequential()

    # Used to be 128 Neurons
    model.add(layers.LSTM(128, return_sequences=True, input_shape=(x_train.shape[1], 3), dropout=0.1))
    #model.add(layers.LSTM(256, input_shape=(x_train.shape[1], 3), dropout=0.1))
    model.add(layers.LSTM(256, return_sequences=True, dropout=0.1))
    
    model.add(layers.LSTM(1024, return_sequences=True, dropout=0.1))

    model.add(layers.LSTM(256, return_sequences=True, dropout=0.1))
    
    model.add(layers.LSTM(32, dropout=0.1))

    model.add(layers.Dense(3))

    model.summary()

    model.compile(optimizer=optimizer,  # Optimizer
                  loss=loss, # Loss function to minimize
                  metrics=metrics # List of metrics to monitor
                  )

    logger.debug("Fit model on training data")
    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=epochs,
                        # We pass some validation for
                        # monitoring validation loss and metrics
                        # at the end of each epoch
                        validation_data=(x_val, y_val))

    scores = model.evaluate(x_train, y_train, verbose=0)
    logger.info("{}: {:.2f}%".format(model.metrics_names[1], scores[1]*100))

    save_model(model)

    # Generate predictions (probabilities -- the output of the last layer)
    # on new data using `predict`
    logger.debug("Generate predictions for samples")
    predictions = model.predict(x_test)
    logger.debug("predictions shape: {}".format(predictions.shape))
    logger.info("predictions: {}".format(predictions[0]))
    logger.info("actual: {}".format(y_test[0]))
    
    print(predictions.shape)
    x_test = x_test.reshape((x_test.shape[0], x_test.shape[1]*x_test.shape[2]))
    print(x_test.shape)
    # invert scaling for forecast
    inv_yhat = np.concatenate((predictions, x_test[:, (3-1)*seq_hist_len:]), axis=1)
    inv_yhat = sc_t.inverse_transform(inv_yhat)
    inv_yhat = inv_yhat[:,:3]
    # invert scaling for actual
    y_test = y_test.reshape((len(y_test), 3))
    inv_y = np.concatenate((y_test, x_test[:, (3-1)*seq_hist_len:]), axis=1)
    inv_y = sc_t.inverse_transform(inv_y)
    inv_y = inv_y[:,:3]
    # calculate RMSE
    rmse = math.sqrt(mean_squared_error(inv_y, inv_yhat))
    print('Test RMSE: %.3f' % rmse)

    plot_predictions(inv_yhat, inv_y)
    plt.show()    
   
    plot_history(history)
    plt.show()

models = {
          "train": train      # Simple RNN
         }

if __name__ == '__main__':
  parser = argparse.ArgumentParser(description='finger_pose_trainer')
  parser.add_argument('--verbose', help='print extra information', action='store_true')
  parser.add_argument('--model', type=str, default="train", help='select which model for training')
  parser.add_argument('--name', type=str, default="", help='set the name to be used when saving the model')
  parser.add_argument('--dataset', type=str, default="all_asllvd.csv", help='dataset to pull from for training the model')
  parser.add_argument('--seqhistlen', type=int, default=30, help='sequence length for sliding window technique?')
  parser.add_argument('--epochs', type=int, default=10, help='number of epochs for training')
  parser.add_argument('--lr', type=float, default=0.0001, help='learning rate for Adam optimizer')

  args = parser.parse_args()

  if args.verbose:
    logger = logging.getLogger("debug")
  else:
    logger = logging.getLogger("basic")

  if models.get(args.model) != None:
    logger.info("Starting training process with {}".format(args.model))
    name = args.name if len(args.name) > 0 else args.model + "_model"
    print(name)
    models.get(args.model)(dataset=args.dataset, seq_hist_len=args.seqhistlen, epochs=args.epochs, optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr))
  else:
    logger.error("{} is not a valid model".format(args.model))

