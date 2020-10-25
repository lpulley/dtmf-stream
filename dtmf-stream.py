"""
A streaming DTMF decoder.
"""

import sounddevice
import numpy
import scipy.signal
from datetime import datetime, timedelta
import time

TONES_LOW = {697, 770, 852, 941}
TONES_HIGH = {1209, 1336, 1477, 1633}
TONES = {
    (697, 1209): '1',
    (697, 1336): '2',
    (697, 1477): '3',
    (697, 1633): 'A',
    (770, 1209): '4',
    (770, 1336): '5',
    (770, 1477): '6',
    (770, 1633): 'B',
    (852, 1209): '7',
    (852, 1336): '8',
    (852, 1477): '9',
    (852, 1633): 'C',
    (941, 1209): '*',
    (941, 1336): '0',
    (941, 1477): '#',
    (941, 1633): 'D',
}

def detect_tone(tone_time, hang_time, *, strength=0.01, debounce=3):
    """
    tone_time: A value in seconds specifying how long a tone must last before it is considered a press
    hang_time: A value in seconds specifying how much silence must follow a press before it is considered complete
    strength: A value in the range [0, 1] specifying how strong a tone must be in order to not be considered noise
    debounce: The number of frames of debouncing to apply; higher values will decrease output noise and responsiveness
    """

    tone_time = timedelta(seconds=tone_time)
    hang_time = timedelta(seconds=hang_time)
    sample_rate = sounddevice.query_devices(sounddevice.default.device, 'input')['default_samplerate']

    last_time = datetime.now() # The last time audio was processed
    saved_tone = None
    seen_tone = None
    seen_count = 0
    time_on = timedelta() # How long ago the current tone was initially seen
    pressed = None # The pressed tone if we are in hang time, otherwise `None`
    complete = False

    def listen(indata, frames, time, status):
        nonlocal last_time, saved_tone, seen_tone, seen_count, time_on, pressed, complete

        # Do a fast Fourier transform to get the magnitudes of different frequencies
        magnitudes = numpy.abs(numpy.fft.rfft(indata[:, 0]))
        # Determine what frequency each bin of `magnitudes` corresponds to
        frequencies = numpy.fft.rfftfreq(frames, 1 / sample_rate)

        # Determine the magnitude for each low and high DTMF tone based on its nearest bin
        magnitudes_low = {tone: magnitudes[numpy.argmin(numpy.abs(frequencies - tone))] for tone in TONES_LOW}
        magnitudes_high = {tone: magnitudes[numpy.argmin(numpy.abs(frequencies - tone))] for tone in TONES_HIGH}

        # Filter out tone magnitudes below the required strength
        magnitudes_low = {tone: magnitude for tone, magnitude in magnitudes_low.items() if magnitude / frames > strength}
        magnitudes_high = {tone: magnitude for tone, magnitude in magnitudes_high.items() if magnitude / frames > strength}

        # Determine the strongest low tone and strongest high tone, and from those, the detected dual-tone
        max_low  = max(magnitudes_low, key=magnitudes_low.get, default=None)
        max_high = max(magnitudes_high, key=magnitudes_high.get, default=None)
        found_tone = TONES.get((max_low, max_high))

        # Keep track of debouncing
        if found_tone == seen_tone:
            if seen_count < debounce:
                # If we haven't seen this tone for long enough, keep counting
                seen_count += 1
            elif found_tone != saved_tone:
                # If we've now see this tone for long enough, save it
                saved_tone = found_tone
                time_on = timedelta()
        else:
            # If the tone just changed, start over
            seen_tone = found_tone
            seen_count = 1
        time_on += datetime.now() - last_time
        last_time = datetime.now()

        # Update the listening state
        if pressed is None:
            if saved_tone is not None and time_on > tone_time:
                # If we have seen a whole tone time elapse, switch to waiting for hang time
                pressed = saved_tone
        else:
            if saved_tone == seen_tone and saved_tone is not None:
                # If we see a tone during hang time, go listen for a new tone
                pressed = None
            elif time_on > hang_time:
                # If we have seen a whole hang time elapse, the press is complete
                complete = True
                raise sounddevice.CallbackStop

        print(last_time, saved_tone, seen_tone, seen_count, time_on, pressed, complete)

    # Create and start a stream, waiting to close it until a press has been detected
    with (stream := sounddevice.InputStream(channels=1, callback=listen, device=sounddevice.default.device)):
        while not complete:
            pass

    return pressed

if __name__ == '__main__':
    while True:
        print(detect_tone(0.25, 0.05), end='', flush=True)