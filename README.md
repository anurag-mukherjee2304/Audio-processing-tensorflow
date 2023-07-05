# Audio-processing-tensorflow

Audio processing can be done in a variety of ways. When using TensorFlow to do experiments with Artificial Neural Networks using audio inputs, the typical workflow is to first preprocess the audio before feeding it to the Neural Net.


# Audio Preprocessing:
When developing a Speech Recognition engine using Deep Neural Networks we need to feed the audio to our Neural Network.

2 common ways to represent sound:

●	Time domain: each sample represents the variation in air pressure.
●	Frequency domain: at each time stamp we indicate the amplitude for each frequency.

# The Spectrogram and the Short Time Fourier Transform :
A spectrogram shows how the frequency content of a signal changes over time and can be calculated from the time domain signal.
The operation, or transformation, used to do that is known as the Short Time Fourier Transform.
We used the example of developing an Automatic Speech Recognition engine, but using the spectrogram as input to Deep Neural Nets is common for non-speech audio tasks such as noise reduction, music genre classification, whale call detection, and so on.


# Discrete Fourier Transform:
For computing the DFT(Discrete Fourier Transform) to evaluate the frequency content of a signal refer:
Theory:
Fourier analysis is basically a method of expressing a function as the sum of periodic components and restoring the function from those components. When both the function and its Fourier transform are replaced by a discretized counterpart, it is called the Discrete Fourier Transform (DFT).
Given a vector of n input amplitudes like:
{ x[0],x[1]...........x[n-1] }

The DFT is defined by the equation:
x[k]=(N-1)summation(n=0) xne(-j2pikn/N)
 
●	k is used to represent the ordinal number of the frequency domain  
●	 n is used to represent the  ordinal number of the time domain 
●	 N is the length of the sequence to be converted.

# Fast Fourier transform:
The Fast Fourier Transform is an efficient implementation of the DFT equation. The signal must be constrained to be  a power of two. 
 This explains why N (the magnitude of the signal at the input of the DFT function) must be a power of 2, otherwise it should be padded with zeros. 
 It's very easy to see if x is a power of two in Python:-

def is_power2(x):
    return x > 0 and ((x & (x - 1)) == 0)

A true sine wave can be expressed as a sum of complex sine waves using Euler's equation.
Since the DFT is a linear function, the total DFT of the sine waves is the sum of the DFTs of each sine wave. Therefore, in the case of the spectrum, we get two DFTs. One for  positive frequencies and the other for  negative frequencies. These are symmetric.

Windowing:
Clipping a signal in the time domain causes ripples  in the frequency domain. 
This can be understood by considering clipping the signal as if applying a rectangular window. Applying a window in the time domain causes a convolution in the frequency domain. 
Ripple occurs when we convolve two frequency domain plots.

Implementation of windowing in python:

from scipy.signal import hanning
import tensorflow as tf
import numpy as np

N = 256 # FFT size
audio = np.random.rand(N, 1) * 2 - 1
w = hanning(N)

input  = tf.placeholder(tf.float32, shape=(N, 1))
window = tf.placeholder(tf.float32, shape=(N))
window_norm = tf.div(window, tf.reduce_sum(window))
windowed_input = tf.multiply(input, window_norm)

with tf.Session() as sess:
    tf.global_variables_initializer().run()
    windowed_input_val = sess.run(windowed_input, {
        window: w,
        input: audio
    })

Zero-Phase Padding:
To use the FFT, the length of the input signal must be a power of two. If the input signal is not of the correct length, we can add leading and trailing zeros  to the signal itself. 
 Since the null sample is originally in the center of the input signal, divide the padded signal by the center and swap the order of these two part:
Zero-Phase

For FFT, magnitude: fft.py

# Short Time Fourier Transform:
The STFT is used to analyze the frequency content of signals when that frequency content varies over time. It can be done by:-

●	Taking segments of the signal.
●	Windowing those out from the rest of the signal, and applying a DFT to each segment.
●	Sliding this window along each segment.
DFT coefficients are obtained as a function of both time and frequency.
The entire code is divided into two files: helpers.py and stft.py .
Refer to the codes above for clear understanding.


Conclusion:

The ability to do the STFT in TensorFlow allows Machine Learning practitioners to transform a signal from the time domain to the frequency domain from any point in the computation graph.
