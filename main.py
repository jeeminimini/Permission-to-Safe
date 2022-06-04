import librosa
import numpy as np
import glob
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import torch
from sklearn.preprocessing import MaxAbsScaler

mel_list = []
sound = glob.glob('C:\\Users\\USER\\Desktop\\sample\\*.wav',recursive=True)
i = 1

for fname in sound:

    i += 1
    audio_signal, sample_rate = librosa.load(fname, duration=10, sr=48000)
    # print(len(audio_signal))
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
    mel_list.append(mel_spec_db)
    # print(mel_spec_db.shape)

    if (i % 100 == 0):
        print(i)

########################
from sklearn.preprocessing import StandardScaler

x_train = np.expand_dims(mel_list, 1) #DataNum, 1ch, H, W

scaler = StandardScaler()

b,c,h,w = x_train.shape
x_train = np.reshape(x_train, newshape=(b,-1))
x_train = scaler.fit_transform(x_train)
x_train = np.reshape(x_train, newshape=(b,c,h,w))
x_train = np.reshape(x_train, newshape=(b,h,w,c))

import joblib
# 객체를 pickled binary file 형태로 저장한다
file_name = 'scaler.pkl'
joblib.dump(scaler, file_name)

#########################

import json
import glob

label_list = []
i = 1
label = glob.glob('C:\\Users\\USER\\Desktop\\sample_json\\*.json',recursive=True)
for label_path in label:
    with open(label_path,'r',encoding="UTF-8") as f:
        json_data = json.load(f)
    #print(json.dumps(json_data,indent = "\t",ensure_ascii=False))

    One = json_data['annotations'][0]['categories']['category_02']
    label_list.append(One)
    i+=1
    if (i%100 == 0):
        print(i)

#################################

y_data = np.array(label_list)

################################

y_data[y_data=='실외'] = 1
y_data[y_data=='실내'] = 1
y_data[y_data!='1'] = 0
y_data = y_data.astype(int)
print(y_data)

######################################

x_train, x_test, y_train, y_test = train_test_split(x_train, y_data, test_size=0.33, random_state=42, stratify = y_data)

######################################

import random

random.seed(1)

x_train = x_train.astype('float')
x_test = x_test.astype('float')
y_train = y_train.astype('int')
y_test = y_test.astype('int')

#############################

# parameters
batch_size = 100
epoch = 5
lr = 0.00005
p = 0.2

import keras
import tensorflow

model = keras.models.Sequential()

model.add(keras.layers.Activation('relu'))
model.add(keras.layers.Conv2D(16,
                   kernel_size=3,
                   strides=(1,1),
                   padding="same",
                   input_shape = (128, 1876, 1)
                  ))
model.add(keras.layers.BatchNormalization(axis = 1))
model.add(keras.layers.MaxPooling2D(pool_size=2, strides=(2,2)))
model.add(keras.layers.Dropout(rate=0.3))
model.add(keras.layers.Conv2D(32,
                   kernel_size=3,
                   strides=(1,1),
                   padding="same"
                  ))
model.add(keras.layers.BatchNormalization(axis = 1))
model.add(keras.layers.MaxPooling2D(pool_size=4, strides=(4,4)))
model.add(keras.layers.Conv2D(64,
                   kernel_size=3,
                   strides=(1,1),
                   padding="same"
                  ))
model.add(keras.layers.BatchNormalization(axis = 1))
model.add(keras.layers.Conv2D(128,
                   kernel_size=3,
                   strides=(1,1),
                   padding="same"
                  ))
model.add(keras.layers.BatchNormalization(axis = 1))
model.add(keras.layers.Flatten())
model.add(keras.layers.Dense(1, use_bias=True,kernel_initializer=keras.initializers.lecun_normal()))
model.add(keras.layers.Activation('sigmoid'))

#############################

model.compile(loss='binary_crossentropy',optimizer='Adam',metrics=['accuracy'])

model.fit(x_train, y_train, batch_size=100, epochs=2)
scores = model.evaluate(x_test,y_test)
print("score : ", scores)

y_pred = model.predict(x_test)
print("y_pred = ",y_pred)
print("answer = ",y_test)
pred = y_pred.reshape((1,-1))
pred.astype('int')

print(pred == y_test)

model.summary()

from keras.models import load_model
model.save('model.h5')
###############################

