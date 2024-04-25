import numpy as np
import matplotlib.pyplot as plt

# Generating the signal
t = np.linspace(0, 1, 1000, endpoint=False)  # Time axis
f1 = 5  # Frequency of the first component
f2 = 20  # Frequency of the second component
signal = np.sin(2 * np.pi * f1 * t) + 0.5 * np.sin(2 * np.pi * f2 * t)

# Fourier transform
fourier_transform = np.fft.fft(signal)
frequencies = np.fft.fftfreq(len(signal), t[1] - t[0])

# Visualization
plt.figure(figsize=(10, 6))

plt.subplot(2, 1, 1)
plt.plot(t, signal)
plt.title('Original Signal')
plt.xlabel('Time')
plt.ylabel('Amplitude')

plt.subplot(2, 1, 2)
plt.plot(frequencies, np.abs(fourier_transform))
plt.title('Fourier Transform')
plt.xlabel('Frequency')
plt.ylabel('Amplitude')

plt.tight_layout()
plt.show()

