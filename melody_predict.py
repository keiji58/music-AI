#必要なモジュールのインポート１
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, LSTM
from keras.optimizers import RMSprop

import numpy as np
import random
import sys

from tqdm import tqdm
import music21 as m21
import os
# import melody_model

path = "C:\\Users\\keiji goya\\Desktop\\KGmusic\\music\\YoasobiTrackList.txt" #トラック指定のファイル

music_keys = ('C')
text = []

TrackList = []
with open(path) as f:
    TrackList = [s.strip().split("=") for s in f.readlines()] #パスとトラック番号を区切ってリスト化


for x in tqdm(TrackList):
    track = m21.converter.parse(x[0])
    print(x[0]+str(len(track.parts)))
    piece = track.parts[int(x[1])-1]
    
    for trans_key in music_keys:
        k = piece.analyze('key')
        trans = trans_key
        
        i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(trans))
        trans_piece = piece.transpose(i)
        for n in trans_piece.flat.notesAndRests:
            if not isinstance(n, m21.chord.Chord):
                text.append(str(n.name) + '_' + str(n.duration.quarterLength) + ' ')



print("LSTM start")
chars = text
count = 0
char_indices = {} #辞書
indices_char = {} #逆引き辞書

for word in chars:
    if not word in char_indices:
        char_indices[word] = count #key=word, value=count
        count += 1
        #print(count, word)

#逆引き辞書の作成
indices_char = dict([(value, key) for (key, value) in char_indices.items()])


maxlen = 16
step = 1
sentences = []
next_chars = []
s = []
for i in range(0, len(text)-maxlen, step): #初項0, 末項len..., 公差step
    sentences.append(text[i:i+maxlen])
    next_chars.append(text[i+maxlen])

print("nb sequences:", len(sentences))

print("Vectorization...")
input_data = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
output_data = np.zeros((len(sentences), len(chars)), dtype=np.bool)

for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        input_data[i, t, char_indices[char]] = 1 #sentence番号, sentence内のindex, 辞書での項目番号
    output_data[i, char_indices[next_chars[i]]] = 1 #正解データ


# 主旋律の生成
def sample(preds, temperature=1.0):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds) #softmax関数の計算
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas) #1になっている要素番号を返す


def make_melody(length=1000):
    #適当にスタートの文を選ぶ
    model = load_model('melo.h5')
    start_index = random.randint(0, len(text)-maxlen-1)
    for diversity in [0.2]:
        generated = ''
        sentence = text[start_index:start_index+maxlen] #ここでユーザの主旋律を与えると続きを生成
        generated += ''.join(sentence) #sentence(list)の各要素を結合してgeneratedに追加
        print(sentence)
        
        for i in range(length):
            input_pred = np.zeros((1, maxlen, len(chars)))
            for t, char in enumerate(sentence):
                input_pred[0, t, char_indices[char]] = 1
            
            preds = model.predict(input_pred, verbose=0)[0] #verbose:詳細 0で詳細情報を表示しない
            next_index = sample(preds, diversity)
            next_char = indices_char[next_index]
            
            generated += next_char
            sentence = sentence[1:]
            sentence.append(next_char)
            
            sys.stdout.write(next_char)
            sys.stdout.flush()
        print()
    return generated


def melody():
    melo_sentence = make_melody(300)
    print(melo_sentence)

    # 出力
    #文字列をmidiに変換
    meas = m21.stream.Stream()
    meas.append(m21.meter.TimeSignature('4/4'))

    meloindex = melo_sentence.split()
    longrest = []
    for index, m in enumerate(meloindex):
        pitch, dist = m.split('_')
        tmp = dist.split('/')
        if len(tmp) == 2: # 分数は少数へ変換
            fdist = float(tmp[0])/float(tmp[1])
        else:
            fdist = float(tmp[0])
        if pitch == "rest":
            if fdist >= 5.0:
                longrest.append(index)

    melo = melo_sentence.split()
    for i in range(len(longrest)):
        del melo[longrest[i] - i] # 長い休符を削除

    for m in melo:
        pitch, dist = m.split('_')
        tmp = dist.split('/')
        if len(tmp) == 2: # 分数は少数へ変換
            fdist = float(tmp[0])/float(tmp[1])
        else:
            fdist = float(tmp[0])
        if pitch == "rest":
            n = m21.note.Rest(quarterLength = fdist) # restはpitchが無い
        else:
            n = m21.note.Note(pitch, quarterLength = fdist)
        meas.append(n)


    meas.makeMeasures(inPlace=True)
    meas.show("midi", addEndTimes=True)
    meas.write("midi", fp="C:\\Users\\keiji goya\\Desktop\\KGmusic\\new_music\\test.mid")
