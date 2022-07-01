import re
import json
import numpy as np
import string
from string import digits

def preproc(text):
    text = text.lower()
    exclude = set(string.punctuation)
    text = ''.join(ch for ch in text if ch not in exclude)
    remove_digits = str.maketrans('', '', digits)
    text = text.translate(remove_digits)
    text = text.strip()
    text = re.sub(" +", " ", text)
    return text

def save_vocab(sorted_words, filename):
    with open(filename, 'w') as f:
        f.writelines([w + '\n' for w in sorted_words])

def load_vocab(filename):
    with open(filename) as f:
        fwd = {word.strip(): i+1 for i, word in enumerate(f)}
    bkw = {i: word for word, i in fwd.items()}
    return fwd, bkw

def load_meta(filename):
    with open(filename) as f:
        meta = json.load(f)
    max_length_src = meta['max_length_source']
    max_length_tar = meta['max_length_target']
    return max_length_src, max_length_tar
