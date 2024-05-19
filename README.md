# Auto Regression Model for Network Music Perfromance Audio Packet Loss Concealment
(ARMNMPAPLC:)

2024
spiral.ok.ubc.ca

An auto regression model for generating audio samples given a training signal.

Takes audio file as input and outputs a generated audio file

# Usage: 
python arPlc.py wavFile.wav
## Wave file format is a mono 44100Hz 16bit

## In the execute def
train = song.getFrames(0, 88200)
## from starting point in seonds, how many samples in the training


ar = AR(22500)
## the order (lag) of the ar model
## larger lag captures temporal changes but increases training time

test = song.getFrames(1, 1024)
## same as above but for the test to generate samples after

res = ar.predict(np.array(test), 44100, 1)
## how many samples to generate, monte carlo sim depth (the MC sim blows out samples, to is disabled in code-leave as one)
