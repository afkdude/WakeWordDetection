import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
import tensorflow as tf

fs = 16000
seconds = 1
filename = "prediction.wav"

interpreter = tf.lite.Interpreter(
    model_path="models/tf_lite_quant_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


def decode_audio(audio_binary):
  audio, _ = tf.audio.decode_wav(contents=audio_binary)
  return tf.squeeze(audio, axis=-1)


def get_spectrogram(waveform):
  input_len = 16000
  waveform = waveform[:input_len]
  zero_padding = tf.zeros(
      [16000] - tf.shape(waveform),
      dtype=tf.float32)
  waveform = tf.cast(waveform, dtype=tf.float32)
  equal_length = tf.concat([waveform, zero_padding], 0)
  spectrogram = tf.signal.stft(
      equal_length, frame_length=255, frame_step=128)
  spectrogram = tf.abs(spectrogram)
  spectrogram = spectrogram[..., tf.newaxis]
  return spectrogram

print("Prediction Started: ")
while True:
    print("Say Now: ")
    myrecording = sd.rec(int(seconds * fs), samplerate=fs,
                         channels=1, dtype='int16')
    sd.wait()
    write(filename, fs, myrecording)
    
    audio_binary = tf.io.read_file(filename)
    waveform = decode_audio(audio_binary)
    spectrogram = get_spectrogram(waveform)
    interpreter.set_tensor(
        input_details[0]['index'], np.array(spectrogram).reshape(1,124,129,1))
    interpreter.invoke()
    tflite_model_predictions = interpreter.get_tensor(
        output_details[0]['index'])

    if np.argmax(tflite_model_predictions,axis=1) >= 6:
        print(f"Wake Word Detected")

    else:
        print(f"Wake Word NOT Detected")
