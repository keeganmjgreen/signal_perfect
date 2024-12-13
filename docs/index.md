# Signal*Per$\!\!\:f\!\!\:$ec$[t]$*

**Resample any signal without compromise.**

SignalPerfect is a Python library for resampling signals.

The signal must be described in a block-based fashion, i.e., split up into intervals (blocks) over each of which the average value of the signal is recorded.

A good example of this is a solar farm's power output --- a value that may continuously fluctuate and vary with time, but may be recorded as the average solar output over 15-minute time blocks. But what if we wanted to upsample this signal to be every minute, or downsample it to 1-hour blocks?

Such resampling problems are often tackled with approaches that are needlessly specific to the application at hand. In the case of upsampling to every minute, one might use linear interpolation (or higher-order spline interpolation). In the case of downsampling to 1-hour blocks, one might bin the 15-minute blocks into the 1-hour blocks and take the average of the four binned values. But what if we wanted to upsample the signal to 10-minute blocks, or downsample it to 20-minute blocks? What if the resampled time blocks do not quite align with the original time blocks? What if the frequency of the input data varies?

Different resampling problems such as these need not be attempted using different approaches --- they can all be solved using the same method, implemented by SignalPerfect.

<br><br><br><br><br> <!-- Add some whitespace. -->