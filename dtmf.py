"""
A streaming DTMF decoder.
"""

import sounddevice
import numpy
import scipy.signal
from datetime import datetime, timedelta

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

def detect_tone(strength=0.01, tone_time=0.5, hang_time=0.1):
    """
    strength: A value in the range [0, 1] specifying how strong a tone must be in order to not be considered noise
    tone_time: A value in seconds specifying how long a tone must last before it is considered a press
    hang_time: A value in seconds specifying how much silence must follow a press before it is considered complete
    """

    tone_time = timedelta(seconds=tone_time)
    hang_time = timedelta(seconds=hang_time)
    sample_rate = sounddevice.query_devices(sounddevice.default.device, 'input')['default_samplerate']

    last_time = datetime.now()
    tone = None
    time_on = timedelta()
    pressed = None
    complete = False

    def callback(indata, frames, time, status):
        nonlocal tone, time_on, pressed, complete

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

        if found_tone == tone:
            time_on += datetime.now() - last_time
        else:
            tone = found_tone
            time_on = timedelta()

        if pressed is None:
            if tone is not None and time_on > tone_time:
                pressed = tone
        else:
            if tone is not None:
                pressed = None
            elif time_on > hang_time:
                complete = True

    with sounddevice.InputStream(channels=1, callback=callback, device=sounddevice.default.device):
        while not complete:
            pass

    return pressed

if __name__ == '__main__':
    while True:
        print(detect_tone(), end='', flush=True)