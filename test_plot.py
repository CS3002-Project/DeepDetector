import numpy as np

import matplotlib.pyplot as plot

 

# Get x values of the sine wave

time        = np.arange(0, 10, 0.1);

 

# Amplitude of the sine wave is sine of a variable like time

amplitude1   = np.sin(time)
amplitude2   = np.sin(time) * 2

args = [time, amplitude1, time, amplitude2]

# Plot a sine wave using time and amplitude obtained for the sine wave

plot.plot(*args)

 

# Give a title for the sine wave plot

plot.title('Label = 1')

 

# Give x axis label for the sine wave plot

plot.xlabel('Time')

 

# Give y axis label for the sine wave plot

plot.ylabel('Readings')
plot.legend(["a1", "a2"])


 

plot.grid(True, which='both')

 

plot.axhline(y=0, color='k')

 

plot.show()

 

# Display the sine wave

plot.show()