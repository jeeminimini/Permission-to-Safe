## 통신 시작
from socket import *
HOST = ""
PORT = 12000
s = socket(AF_INET, SOCK_STREAM)
print ('Socket created')
s.bind((HOST, PORT))
print ('Socket bind complete')
s.listen(1)
print ('Socket now listening')

conn, addr = s.accept()
print("Connected by ", addr)

# 현재 상황 소리 녹음 후 저장
import pyaudio
import wave

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
RECORD_SECONDS = 5

audio = pyaudio.PyAudio()

stream = audio.open(format=FORMAT,
       channels=CHANNELS,
       rate=RATE,
       input=True,
       frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE/CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")


# stop Recording

stream.stop_stream()
stream.close()
audio.terminate()

WAVE_OUTPUT_FILENAME = "file.wav"
waveFile = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
waveFile.setnchannels(CHANNELS)
waveFile.setsampwidth(audio.get_sample_size(FORMAT))
waveFile.setframerate(RATE)
waveFile.writeframes(b''.join(frames))
waveFile.close()

## 저장했던 인공지능 모델 불러오기
from keras.models import load_model
model = load_model('C:\\Users\\USER\\PycharmProjects\\pythonProject\\Project_Final\\model.h5')

## 학습 시 사용했던 scaler 불러오기
import joblib
file_name = 'C:\\Users\\USER\\PycharmProjects\\pythonProject\\AI_model\\scaler.pkl'
scaler = joblib.load(file_name)

model.summary()

# 녹음된 음성 불러오기
import tensorflow
#path  = "C:\\Users\\USER\\PycharmProjects\\pythonProject\\Project_Final"
#fname = "\\file.wav"
path = "C:\\Users\\USER\\Desktop\\sample"
fname = "\\17.실내_906904_label.wav"

# 녹음된 음성 데이터 전처리
import librosa
import numpy as np

audio_signal, sample_rate = librosa.load(path+fname, duration=10, sr=48000)
signal = np.zeros(int(48000 * 10 + 1, ))
signal[:len(audio_signal)] = audio_signal

mel_spec = librosa.feature.melspectrogram(y=signal,
                                          sr=48000,
                                          n_fft=1024,
                                          win_length=512,
                                          window='hamming',
                                          hop_length=256,
                                          n_mels=128,
                                          fmax=sample_rate / 2
                                          )

mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

print(mel_spec_db.shape)
h, w = mel_spec_db.shape
x_train = np.reshape(mel_spec_db, newshape=(1,-1))
print(x_train.shape)

x_train = scaler.transform(x_train)
print(x_train.shape)

x_train = np.reshape(x_train, newshape=(1,1,h,w))
print(x_train.shape)

now = np.reshape(x_train, newshape=(1,h,w,1))
print(now.shape)
print(now)

## 현재 음성 데이터에 대한 예측값 반환
predict = model.predict(now)
print(predict[0][0])

if (predict[0][0]>=0.5):pred=int(1)
else: pred=int(0)

print(pred)

## 현재 상황 예측값을 안드로이드 앱에 통신
while True:
    #데이터 수신
    rc = conn.recv(1024)
    rc = rc.decode("utf8").strip()
    print(rc)
    if rc=="통신중단":
        print("Received: " + rc)
        receive=rc
        #연결 닫기
        conn.close()
        break
    if rc=="통신시작":
        a=1
    if pred==0:
        res = "위급상황"
        print("현재상황 : 위급 : " , res)
    else:
        res = "일반상황"
        print("현재상황 : 일반 : " , res)
    #클라이언트에게 답을 보냄
    conn.sendall(res.encode("utf-8"))
    conn.close()
    break
s.close()


