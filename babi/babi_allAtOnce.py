import tarfile
import babi_parse
import numpy as np
import subprocess

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences




def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return pad_sequences(xs, maxlen=story_maxlen), pad_sequences(xqs, maxlen=query_maxlen), np.array(ys)

def printshape(layer, should_print=True):
    if should_print:
        print(layer.shape)

        
subprocess.check_output('cd tasks; ./make_tasks.sh 1 20', shell=True)

params = {'flatten_sentences': True,
         'only_supporting': False}

with tarfile.open('./tasks/tasks.tar') as tar:
    train = babi_parse.extract_with_token(tar, 'train', flatten_tasks=True, **params)
    test_nested, task_names = babi_parse.extract_with_token(tar, 'test', flatten_tasks=False, **params) # so that we can evaluate performance on each category
    test_flattened = babi_parse.extract_with_token(tar, 'test', flatten_tasks=True, **params)



vocab = set()
for story, q, answer in train + test_flattened:
    vocab |= set(story + q + [answer]) # union operation |=
vocab = sorted(vocab)
len(vocab) # pretty small vocab


vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test_flattened)))
query_maxlen = max(map(len, (x for _, x, _ in train + test_flattened)))
x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test_flattened, word_idx, story_maxlen, query_maxlen)

ttx = []; ttxq = []; tty = [];
for taskid, task in enumerate(test_nested):
    a,b,c = vectorize_stories(task, word_idx, story_maxlen, query_maxlen)
    ttx.append(a); ttxq.append(b); tty.append(c)


RNN = recurrent.LSTM
EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 20
DIAGNOSTIC_PRINT=True
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format(RNN,
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
							   QUERY_HIDDEN_SIZE))

showshape = lambda x: printshape(x, should_print=DIAGNOSTIC_PRINT)

sentence = layers.Input(shape=(story_maxlen,), dtype='int32')
showshape(sentence)
encoded_sentence = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(sentence)
showshape(encoded_sentence)
encoded_sentence = layers.Dropout(0.3)(encoded_sentence)
showshape(encoded_sentence)


question = layers.Input(shape=(query_maxlen,), dtype='int32')
showshape(question)
encoded_question = layers.Embedding(vocab_size, EMBED_HIDDEN_SIZE)(question)
showshape(encoded_question)
encoded_question = layers.Dropout(0.3)(encoded_question)
showshape(encoded_question)
encoded_question = RNN(EMBED_HIDDEN_SIZE)(encoded_question)
showshape(encoded_question)
encoded_question = layers.RepeatVector(story_maxlen)(encoded_question)
showshape(encoded_question)


merged = layers.add([encoded_sentence, encoded_question])
showshape(merged)
merged = RNN(EMBED_HIDDEN_SIZE)(merged)
showshape(merged)
merged = layers.Dropout(0.3)(merged)
showshape(merged)
preds = layers.Dense(vocab_size, activation='softmax')(merged)
showshape(preds)



model = Model([sentence, question], preds)
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

print('Training')
model.fit([x, xq], y,
          batch_size=BATCH_SIZE,
          epochs=EPOCHS,
          validation_split=0.05,
          shuffle=True)
loss, acc = model.evaluate([tx, txq], ty,
                           batch_size=BATCH_SIZE)
print('Test loss / test accuracy = {:.4f} / {:.4f}'.format(loss, acc))

for i in range(20):
    loss, acc = model.evaluate([ttx[i], ttxq[i]], tty[i])
    print('Task:\t{}\tAccuracy:\t{}'.format(i+1, acc))






