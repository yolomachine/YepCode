import os

corpus = set()
path = os.path.join('revisited', 'java', 'full', 'stm.corpus')
with open(path, 'r', encoding='utf-8') as fp:
    for i in fp.read().strip().split('\n'):
        corpus.add(i)

with open(path, 'w', encoding='utf-8') as fp:
    fp.write('\n'.join(list(corpus)))
