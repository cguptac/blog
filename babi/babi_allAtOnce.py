from functools import reduce
import re
import tarfile

import numpy as np

from keras.utils.data_utils import get_file
from keras.layers.embeddings import Embedding
from keras import layers
from keras.layers import recurrent
from keras.models import Model
from keras.preprocessing.sequence import pad_sequences

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split('(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format
    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            substory = None
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data

def get_stories(f, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data if not max_length or len(flatten(story)) < max_length]
    return data


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


# get all the files with a given token, train or test
def extract_with_token(tar, token, flatten=True):
    files = []
    flat_file = []
    task_names = []
    for member in tar.getmembers():
        if token in member.name:
            tar.extract(member, './')
            with open(member.name) as infile:
                new_stories = get_stories(infile)
            if flatten:
                for story in new_stories:
                    flat_file.append(story)
            else:
                files.append(new_stories)
                task_names.append([member.name])
    if flatten:
        return flat_file
    else:
        return files, task_names




with tarfile.open('./tasks/tasks.tar') as tar:
    train = extract_with_token(tar, 'train', flatten=True)
    test_nested, task_names = extract_with_token(tar, 'test', flatten=False) # so that we can evaluate performance on each category
    test_flattened = extract_with_token(tar, 'test', flatten=True)
#len(train), len(test_nested), len(test_flattened), sum(len(_) for _ in test_nested)

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






