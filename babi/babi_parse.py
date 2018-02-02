from functools import reduce
import re
import tarfile
from collections import namedtuple
import itertools

# story, question, answer, relevant
Sqar = namedtuple('StoryQuestionAnswerRelevant', ['data', 'question', 'answer', 'relevant'])
# story, question, answer
Sqa = namedtuple('StoryQuestionAnswer', ['data', 'question', 'answer'])



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
            supporting = [int(_) for _ in supporting.split()]
            if only_supporting:
                # Only select the related substory               
                substory = [story[i - 1] for i in supporting]
                supporting=None
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a, supporting))
            story.append(q)
        else:
            sent = tokenize(line)
            story.append(sent)
    return data




def get_stories(f, max_length=None, **kwargs):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.
    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    only_supporting = kwargs.get('only_supporting', False)
    flatten_sentences = kwargs.get('flatten_sentences', False)
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    if flatten_sentences:
        data = [Sqa(flatten(story), q, answer) \
                for story, q, answer, supporting in data if not max_length or len(flatten(story)) < max_length]
    else:
        data = [Sqar(story, q, answer, supporting) \
               for story, q, answer, supporting in data if not max_length or len(story) < max_length]
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
def extract_with_token(tar, token, **kwargs):
    flatten_tasks = kwargs.get('flatten_tasks', True)
    files = []
    flat_file = []
    task_names = []
    for member in tar.getmembers():
        if token in member.name:
            tar.extract(member, './')
            with open(member.name) as infile:
                new_stories = get_stories(infile, **kwargs)
            if flatten_tasks:
                for story in new_stories:
                    flat_file.append(story)
            else:
                files.append(new_stories)
                task_names.append([member.name])
    if flatten_tasks:
        return flat_file
    else:
        return files, task_names

    
def flatten_nested(nested_list):
    return list(itertools.chain.from_iterable(nested_list))

