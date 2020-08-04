import wave
import sys
from pydub import AudioSegment
import soundfile as sf
import pyrubberband as pyrb

# sound = AudioSegment.from_mp3(sys.argv[1])
# sound.export("file.wav", format="wav")

y, sr = sf.read("0.wav")
y_stretch = pyrb.time_stretch(y, sr, 0.90)
y_shift = pyrb.pitch_shift(y, sr, 0.90)
sf.write("analyzed_filepathX5.wav", y_stretch, sr, format='wav')