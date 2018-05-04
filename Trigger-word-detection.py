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


__,,  datadata  ==  wavfilewavfile.read("audio_examples/example_train.wav")
print("Time steps in audio recording before spectrogram", data[:,0].shape)
print("Time steps in input after spectrogram", x.shape)