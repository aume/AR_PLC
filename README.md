# Auto Regression Model for Network Music Perfromance Audio Packet Loss Concealment
(ARMNMPAPLC:)

2024
spiral.ok.ubc.ca
An auto regression model for generating audio samples given a training signal.
Takes audio file as input and outputs a generated audio file

# Requires:
numpy
sklearn

# Usage: 
## Wave file format is a mono 44100Hz 16bit
python arPlc.py wavFile.wav


## In the execute def
## from starting point in seonds, how many samples in the training
train = song.getFrames(0, 88200)




## the order (lag) of the ar model. larger lag captures temporal changes but increases training time
ar = AR(22500)

## same as above but for the test to generate samples after
test = song.getFrames(1, 1024)

## how many samples to generate, monte carlo sim depth (the MC sim blows out samples, to is disabled in code-leave as one)
res = ar.predict(np.array(test), 44100, 1)

