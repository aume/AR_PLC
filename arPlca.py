'''
Auto Regression Model for Network Music Perfromance Audio Packet Loss Concealment
(ARMNMPAPLC:)

2024
spiral.ok.ubc.ca

An auto regression model for generating audio samples given a training signal.

Takes audio file as input and outputs a generated audio file

Usage: 
python arPlc.py wavFile.wav
# Wave file format is a mono 44100Hz 16bit

# In the execute def
train = song.getFrames(0, 44100)
# from starting point in seonds, how many samples in the training


ar = AR(512)
# the order (lag) of the ar model
# larger lag captures temporal changes but increases training time

test = song.getFrames(1, 1024)
# same as above but for the test to generate samples after

res = ar.predict(np.array(test), 2048, 1)
# how many samples to generate, monte carlo sim depth (the MC sim blows out samples, to is disabled in code-leave as one)

'''


import wave
import struct
from sklearn.linear_model import LinearRegression
import sys
import numpy as np



def execute():
    chunk = 0.1 * 1000 # in seconds1024
    # open the file for reading.
    song = WAVrw(sys.argv[1])
    # read data (based on the chunk size)
    train = song.getFrames(0, 44100)
    
    ar = AR(512)
    
    ar.fit(np.array(train))

    test = song.getFrames(1, 1024)
    #print (np.array(test))
    res = ar.predict(np.array(test), 2048, 1)
    res = res.astype(int)
    res = np.squeeze(res).tolist()
    print(train)

    song.write_wav_file(sys.argv[1]+'.wav', res)


# adapted from https://medium.com/@karen.mossoyan/building-an-autoregression-model-for-time-series-analysis-in-python-49402bdd6d08
class AR:
  def __init__(self, p):
    '''p is the order of the model'''
    self.p = p
    self.model = LinearRegression()
    self.sigma = None

  def generate_train_x(self, X):
    n = len(X)
    ans = X[:n-self.p] # get first column data
    ans = np.reshape(ans, (-1, 1)) # columnise it
    for k in range(1, self.p):
      temp = X[k:n-self.p+k] # get following column offset by k
      temp = np.reshape(temp, (-1, 1)) # columnise it
      ans = np.hstack((ans, temp)) # concatenate
    return ans
  
  def generate_train_y(self, X):
    # get all Xs offset by p
    return X[self.p:]

  def fit(self, X):
    self.sigma = np.std(X)
    train_x = self.generate_train_x(X)
    train_y = self.generate_train_y(X)
    self.model.fit(train_x, train_y)

  def predict(self, X, num_predictions, mc_depth):
    '''
    Here the input parameters include num_predictions and mc_depth. 
    Num_predictions is a variable which stands for the amount of steps which we will be 
    predicting based on the given data. Mc_depth is the amount of Monte Carlo simulations 
    of the model we will be averaging.
    The way this function works is, it takes the last p values of the given signal 
    (variable a in the function), and predicts the next value using the trained model and noise. 
    Then the set of values a drops the oldest value and puts the newest value at its front.
    We continue this way until we predict the desired number of steps.
    This process defines one cycle of the Monte Carlo simulation. 
    So we will repeat it over and over until we get the desired amount of simulations.
    '''
    X = np.array(X)
    ans = np.array([])

    for j in range(mc_depth):
      ans_temp = []
      a = X[-self.p:]
      for i in range(num_predictions):
        # passing on the monte carlo sims as addative std blows out audio samples
        next = self.model.predict(np.reshape(a, (1, -1)))# + np.random.normal(loc=0, scale=self.sigma)

        ans_temp.append(next)
        
        a = np.roll(a, -1)
        a[-1] = next
      
      if j==0:
        ans = np.array(ans_temp)
      
      else:
        ans += np.array(ans_temp)
    
    ans /= mc_depth
    return ans
  
  def score(self, X):
    train_x = self.generate_train_x(X)
    train_y = self.generate_train_y(X)
    return self.model.score(train_x, train_y)
  
class WAVrw:
  def __init__(self, wFile):
    self.samples, self.sample_width, self.num_channels, self.frame_rate = self.read_wav_file(wFile)
  
  def read_wav_file(self, file_path):
    # Open the WAV file
    with wave.open(file_path, 'rb') as wav_file:
        # Get the number of frames in the file
        num_frames = wav_file.getnframes()
        # Read all frames
        frames = wav_file.readframes(num_frames)
        # Get the number of channels (e.g., 1 for mono, 2 for stereo)
        num_channels = wav_file.getnchannels()
        # Get the sample width in bytes
        sample_width = wav_file.getsampwidth()
        # Get the frame rate (samples per second)
        frame_rate = wav_file.getframerate()
        
        print(f'Number of frames: {num_frames}')
        print(f'Number of channels: {num_channels}')
        print(f'Sample width: {sample_width}')
        print(f'Frame rate: {frame_rate}')
        
        # Determine the format string for unpacking the frames
        if sample_width == 1:
            fmt = f'{num_frames * num_channels}B'  # 8-bit audio
        elif sample_width == 2:
            fmt = f'{num_frames * num_channels}h'  # 16-bit audio
        else:
            raise ValueError(f'Unsupported sample width: {sample_width}')
        
        # Unpack the binary data into integers
        samples = struct.unpack(fmt, frames)
        
        # If stereo, separate the channels
        if num_channels == 2:
            left_channel = samples[0::2]
            #right_channel = samples[1::2]
            return left_channel, sample_width, num_channels, frame_rate
        else:
            return samples, sample_width, num_channels, frame_rate 

  def getFrames(self, pos, nFrames):
      # Select a section of the samples (e.g., from second 1 to second 2)
    start_frame = self.frame_rate * 1
    end_frame = start_frame + nFrames #self.frame_rate * 2
    if self.num_channels == 2:
        left_channel_section = self.samples[0][start_frame:end_frame]
        right_channel_section = self.samples[1][start_frame:end_frame]
        #return (left_channel_section, right_channel_section)
        return left_channel_section
    else:
        return self.samples[start_frame:end_frame]
    

  def write_wav_file(self, file_path, samples):
    
    with wave.open(file_path, 'wb') as wav_file:
        wav_file.setnchannels(self.num_channels)
        wav_file.setsampwidth(self.sample_width)
        wav_file.setframerate(self.frame_rate)
        
        # Determine the format string for packing the frames
        if self.sample_width == 1:
            fmt = 'B'  # 8-bit audio
        elif self.sample_width == 2:
            fmt = 'h'  # 16-bit audio
        else:
            raise ValueError(f'Unsupported sample width: {self.sample_width}')
        
        # If stereo, interleave the channels
        if self.num_channels == 2:
            interleaved = []
            for l, r in zip(samples[0], samples[1]):
                interleaved.append(l)
                interleaved.append(r)
            packed_data = struct.pack(f'<{len(interleaved)}{fmt}', *interleaved)
        else:
            packed_data = struct.pack(f'<{len(samples)}{fmt}', *samples)
        
        wav_file.writeframes(packed_data)

execute()
