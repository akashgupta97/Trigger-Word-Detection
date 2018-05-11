import numpy as np
from pydub import AudioSegment
import random
import sys
import io
import os
import glob
import IPython
from td_utils import *
%matplotlib inline


IPython.display.Audio("./raw_data/activates/1.wav")

IPythonIPython.display.Audio("./raw_data/negatives/4.wav")

IPython.display.Audio("./raw_data/backgrounds/1.wav")


IPython.display.Audio("audio_examples/example_train.wav")


x =graph_spectrogram("audio_examples/example_train.wav")


_,data = wavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)

Tx = 5511 # The number of time steps input to the model from the spectrogram
n_freq = 101 # Number of frequencies input to the model at each time step of the spectrogram



Ty =  1375  # The number of time steps in the output of our model

# Load audio segments using pydub
activates, negatives, backgrounds = load_raw_audio()

print("background len: " + str(len(backgrounds[0])))    # Should be 10,000, since it is a 10 sec clip
print("activate[0] len: " + str(len(activates[0])))     # Maybe around 1000, since an "activate" audio clip is usually around 1 sec (but varies a lot)
print("activate[1] len: " + str(len(activates[1])))     # Different "activate" clips can have different lengths

'''
background len: 10000
activate[0] len: 916
activate[1] len: 1579
'''


def get_random_time_segment(segment_ms):
    """
    Gets a random time segment of duration segment_ms in a 10,000 ms audio clip.

    Arguments:
    segment_ms -- the duration of the audio clip in ms ("ms" stands for "milliseconds")

    Returns:
    segment_time -- a tuple of (segment_start, segment_end) in ms
    """


segment_start = np.random.randint(low=0,
                                  high=10000 - segment_ms)  # Make sure segment doesn't run past the 10sec background
segment_end = segment_start + segment_ms - 1

return (segment_start, segment_end)