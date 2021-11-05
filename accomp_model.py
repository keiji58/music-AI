#必要なモジュールのインポート３
from __future__ import print_function
from keras.callbacks import LambdaCallback
from keras.models import Sequential, load_model, Model
from keras.layers import Dense, LSTM, Input
from keras.optimizers import RMSprop, Adadelta, Adam
import numpy as np
import random
import sys
import music21 as m21

# 音符・休符の文章への変換
# トラックの結合
def concatenateTracks(path, trackNums):
    track = m21.converter.parse(path)
    p = m21.stream.Part(id="part")
    for i in trackNums:
        piece = track.parts[int(i)-1]
        for note in piece.flat.notes:
            p.insert(note.offset, note)    
    return p

#今回はいきなり文章を作る
unitlength = 2 #0.5小節
music_keys = ('C')
def TrackToStrList(path, trackNums):
    track = m21.converter.parse(path)
    piece = concatenateTracks(path, trackNums)#コードなので結合
    TrackStr = [""]*(int(piece.quarterLength/unitlength)+1) 
    #1回あたり何要素か
    
    for trans_key in music_keys:
        k = piece.analyze('key')
        trans = trans_key
        
        i = m21.interval.Interval(k.tonic, m21.pitch.Pitch(trans))
        trans_piece = piece.transpose(i)
        
        for n in trans_piece.flat.notes:
            notes = []
            if isinstance(n, m21.note.Note):
                notes = [n]
            elif isinstance(n, m21.chord.Chord):
                notes = [x for x in n]

            offset = n.offset
            for note in notes:
                q = int(offset // unitlength)
                r = offset % unitlength
                TrackStr[q] += note.nameWithOctave + '_' + str(n.duration.quarterLength) + '_' + str(r) + ' '
            #0.5小節(=4分音符2つ)単位で見たとき、先頭からどれくらい離れているか=オフセット
    return TrackStr

# とりあえず１ファイル分だけで(伴奏とメロディの組み合わせのデータ)
InputSentences = TrackToStrList('C:\\Users\\keiji goya\\Desktop\\KGmusic\\music\\yorunikakeru_simple.mid',  ['1'])
# OutputSentences = TrackToStrList('C:\\Users\\keiji goya\\Desktop\\KGmusic\\music\\yorunikakeru_simple.mid',  ['1'])
# OutputSentences = TrackToStrList('C:\\Users\\keiji goya\\Desktop\\KGmusic\\music\\YorunikakeruAccomp.mid', ['1'])
OutputSentences = TrackToStrList('C:\\Users\\keiji goya\\Desktop\\KGmusic\\music\\Ed Sheeran - Shape of You.mid', ['1'])

# 主旋律と伴奏の組を作る
meloStart = 0
while InputSentences[meloStart] == '':
    meloStart += 1 # melostartはInputSentencesの要素数になる
meloEnd = len(InputSentences)-1
while InputSentences[meloEnd] == '':
    meloEnd -= 1

InputSentences = InputSentences[meloStart:meloEnd+1]
OutputSentences = ["\t "+s+"\n" for s in OutputSentences[meloStart:meloEnd+1]]

print(InputSentences)
print(OutputSentences)

# 辞書の作成
#input=メロディ, output=伴奏の2種類を作成
count = 0
input_token_indices = {}
InputChars = ''.join(InputSentences).split()
OutputChars = ''.join(OutputSentences).split()

for word in InputChars:
    if not word+' ' in input_token_indices:
        input_token_indices[word+' '] = count #key=word, value=count
        count += 1
count = 0
output_token_indices = {}

for word in OutputChars:
    if not word+' ' in output_token_indices:
        output_token_indices[word+' '] = count #key=word, value=count
        count += 1
output_token_indices['\t '] = count
output_token_indices['\n'] = count+1

#逆引き辞書の作成
reverse_input_char_index = dict(
    (i, char) for char, i in input_token_indices.items())
reverse_output_char_index = dict(
    (i, char) for char, i in output_token_indices.items())

# One-Hotベクトル化
num_encoder_tokens = len(input_token_indices)
num_decoder_tokens = len(output_token_indices)
encoder_maxlen = max([len(txt.split()) for txt in InputSentences])
decoder_maxlen = max([len(txt.split(' ')) for txt in OutputSentences])
num_sentences = len(InputSentences)
print("nb sequences:", num_sentences)

print("Vectorization...")
encoder_input_data = np.zeros((num_sentences, encoder_maxlen, num_encoder_tokens), dtype='float32')
decoder_input_data = np.zeros((num_sentences, decoder_maxlen, num_decoder_tokens), dtype='float32')
decoder_output_data = np.zeros((num_sentences, decoder_maxlen, num_decoder_tokens), dtype='float32')

for i, (input_text, output_text) in enumerate(zip(InputSentences, OutputSentences)):
    for t, char in enumerate(input_text.split()):
        encoder_input_data[i, t, input_token_indices[char+' ']] = 1.
    for t, char in enumerate(output_text.split(' ')):
        if char != '\n':
            char += ' '
        # decoder_output_dataはタイムステップが1遅れる
        decoder_input_data[i, t, output_token_indices[char]] = 1.
        if t > 0:
            decoder_output_data[i, t - 1, output_token_indices[char]] = 1.

# モデルの作成・学習
print("Build model...")

batch_size = 1
epochs = 100 # ほんとは500
latent_dim = 15 #隠れ層の次元

# エンコーダの作成
encoder_inputs = Input(shape=(None, num_encoder_tokens))
encoder = LSTM(latent_dim, return_state=True)
encoder_outputs, state_h, state_c = encoder(encoder_inputs)
# エンコーダの状態のみ保持 出力は捨てる
encoder_states = [state_h, state_c]

# デコーダの作成
decoder_inputs = Input(shape=(None, num_decoder_tokens))
decoder_lstm = LSTM(latent_dim, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs,initial_state=encoder_states) #初期状態はエンコーダの最終状態
decoder_dense = Dense(num_decoder_tokens, activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
stack = model.fit([encoder_input_data, decoder_input_data], decoder_output_data,batch_size=batch_size,epochs=epochs,validation_split=0.2)
model.save('s2s.h5')


# # 主旋律への伴奏付け
# def sample(preds, temperature=1.0):
#     preds = np.asarray(preds).astype('float64')
#     preds = np.log(preds) / temperature
#     exp_preds = np.exp(preds)
#     preds = exp_preds / np.sum(exp_preds) #softmax関数の計算?
#     probas = np.random.multinomial(1, preds, 1) 
#     #http://www.gentosha-academy.com/serial/okamoto-4/ にmultinomialの説明あり
#     #今回は1からkがpreds=[p1,p2,...,pk]という度数分布に従っているとき、
#     #1回(var1)試行を行ったときの度数分布が1サンプル(var3)得られる
#     return np.argmax(probas) #1になっている要素番号を返す

# 推論用のモデル作成
# encoder_model = Model(encoder_inputs, encoder_states)

# decoder_state_input_h = Input(shape=(latent_dim,))
# decoder_state_input_c = Input(shape=(latent_dim,))
# decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
# decoder_outputs, state_h, state_c = decoder_lstm(
#     decoder_inputs, initial_state=decoder_states_inputs)
# decoder_states = [state_h, state_c]
# decoder_outputs = decoder_dense(decoder_outputs)
# # 推論ではデコーダの状態も使う
# decoder_model = Model(
#     [decoder_inputs] + decoder_states_inputs,
#     [decoder_outputs] + decoder_states)

# def decode_sequence(input_seq):
#     states_value = encoder_model.predict(input_seq)

#     output_seq = np.zeros((1, 1, num_decoder_tokens))
#     output_seq[0, 0, output_token_indices['\t ']] = 1.

#     stop_condition = False
#     decoded_sentence = ''
#     while not stop_condition:
#         output_tokens, h, c = decoder_model.predict(
#             [output_seq] + states_value) 
#         #1回目の実行では、エンコーダ(メロディ)の予測と先頭文字だけのアウトプット(伴奏)をつなげる

#         sampled_token_index = sample(output_tokens[0, -1, :], 0.2)
#         sampled_char = reverse_output_char_index[sampled_token_index]
        
#         #最大長に達するか終端文字が来たら生成終了
#         if sampled_char != '\n':
#             decoded_sentence += sampled_char + ' '
#         else:
#             stop_condition = True
        
#         if len(decoded_sentence.split()) > decoder_maxlen:
#             stop_condition = True

#         # 出力文字の更新
#         output_seq = np.zeros((1, 1, num_decoder_tokens))
#         output_seq[0, 0, sampled_token_index] = 1.

#         # 状態の更新
#         states_value = [h, c]

#     return decoded_sentence



# def accomp():
#     model = load_model('s2s.h5')
#     NewSentences=TrackToStrList("C:\\Users\\keiji goya\\Desktop\\KGmusic\\new_music\\test.mid", ['1'])
#     meloStart = 0
#     while NewSentences[meloStart] == '':
#         meloStart += 1
#     meloEnd = len(NewSentences)-1
#     while NewSentences[meloEnd] == '':
#         meloEnd -= 1
#     NewSentences = NewSentences[meloStart:meloEnd+1]
#     new_num_sentences = len(NewSentences)
#     input_sentence = []
#     output_sentence = []
#     for seq_index in range(new_num_sentences):
#         InputSeq = encoder_input_data[seq_index: seq_index + 1] #encoder_input_dataの先頭0.5小節から順に生成
#         decoded_sentence = decode_sequence(InputSeq)
#         print('-')
#         print('Input sentence:', NewSentences[seq_index])
#         print('Decoded sentence:', decoded_sentence)
#         input_sentence.append(NewSentences[seq_index]) #入力文章を追加 
#         output_sentence.append(decoded_sentence) #生成結果を追加

#     # 出力
#     # 分数をfloatに変換できるようにする
#     from fractions import Fraction
#     #文字列をmidiに変換
#     score = m21.stream.Score()
#     meloPart = m21.stream.Part(id="melo")
#     accompPart = m21.stream.Part(id="accomp")

#     #主旋律のトラック作成
#     melo = input_sentence
#     offset = 0
#     for i, ms in enumerate(melo):
#         for m in ms.split():
#             pitch, length, _offset = m.split('_')

#             tmp = length.split('/')
#             if len(tmp) == 2:
#                 length = float(tmp[0])/float(tmp[1])
#             else:
#                 length = float(tmp[0])

#             offset = float(Fraction(_offset))
#             n = m21.note.Note(pitch, quarterLength=length)
#             meloPart.insert(i*unitlength+offset,n)

#     #伴奏のトラック作成
#     accomp = output_sentence
#     offset = 0
#     for i, ms in enumerate(accomp):
#         for m in ms.split():
#             pitch, length, _offset = m.split('_')

#             tmp = length.split('/')
#             if len(tmp) == 2:
#                 length = float(tmp[0])/float(tmp[1])
#             else:
#                 length = float(tmp[0])

#             offset = float(Fraction(_offset))
#             n = m21.note.Note(pitch, quarterLength=length)
#             accompPart.insert(i*unitlength+offset,n)
            
#     score.insert(0, meloPart)
#     score.insert(0, accompPart)
#     score.show("midi")
#     score.write(fmt="midi", fp="C:\\Users\\keiji goya\\Desktop\\KGmusic\\new_music\\newmusic.mid")