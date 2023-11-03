"""
This is a utility script
"""

# preprocessing functions
import re
import requests
import os
import nltk
import spacy 

from nltk.corpus import stopwords
from nltk.tokenize.casual import TweetTokenizer
from nltk.tokenize import word_tokenize, RegexpTokenizer

from typing import List, Set, Union, Iterable, Dict
from nltk.stem import WordNetLemmatizer

## Vocabulary related functions

def extended_english_dict(dict_file_name = 'dictionary.txt') -> Set:
    dictionary_file_url = 'https://raw.githubusercontent.com/dwyl/english-words/master/words_alpha.txt'

    script_directory = os.path.dirname(os.path.realpath(__file__))
    dict_path = os.path.join(script_directory, dict_file_name) 
    
    if os.path.exists(dict_path):
        print("the dictionary has already been downloaded!! ")
    else:
        r = requests.get(dictionary_file_url, allow_redirects=True)        
        with open(dict_path, 'wb') as f:
            f.write(r.content)

    # process the text file    
    dictionary = set()
    with open(dict_path, 'r') as f:
        for line in f.readlines():        
            word = line[:-1]
            if len(word) >= 3:
                dictionary.add(word)

    return dictionary

def extended_stop_words(stop_words_file_name: str = 'stop_words.txt') -> Set:
    stopwords_file_url = 'https://raw.githubusercontent.com/stopwords-iso/stopwords-en/master/stopwords-en.txt'
    r = requests.get(stopwords_file_url, allow_redirects=True)

    script_directory = os.path.dirname(os.path.realpath(__file__))
    stop_words_path = os.path.join(script_directory, stop_words_file_name) 

    with open(stop_words_path, 'wb') as f:
        f.write(r.content)

    # process the text file    
    stop_words = set()
    with open(stop_words_path, 'r') as f:
        for line in f.readlines():        
            word = line[:-1]
            # ignore characters
            if len(word) >= 2:
                stop_words.add(word)


    return stop_words

def standard_stop_words() -> Set:
    try:
        return set(stopwords.words('english'))
    except OSError:
        nltk.download('stopwords')
        return set(stopwords.words('english'))

# Tokenization functions
def _tweek_tokenize(sentence: str) -> List[str]:
    return TweetTokenizer().tokenize(sentence)

def _word_tokenize(sentence: str) -> List[str]:
    return word_tokenize(sentence)

def _space_tokenize(sentence: str) -> List[str]:
    return RegexpTokenizer('\s+').tokenize(sentence)

def tokenize(sentence: str, tokenizer_type='tweet') -> List[str]:
    if tokenizer_type not in ['tweet', 'space', 'word']:
        raise ValueError(f"Please make sure to pass a supported tokenizer_type: {['twitter', 'space', 'word']}")

    tokenizers = {'tweet': _tweek_tokenize, 'space': _space_tokenize, "word": _word_tokenize}
    return tokenizers[tokenizer_type](sentence)


def filter_text(sentence: str, 
                filter_words: set[str] = None, 
                filter_function: callable = None, 
                tokenizer = None, 
                tokenizer_type: str = 'tweet') -> str:
    # at least 'filler_words' or 'filter_functions' must be passed
    if filter_words is None and filter_function is None:
        raise TypeError(f"At least one of the arguments: 'filter_words' or 'filter_functions' must be passed")

    filter_function = lambda _ : True if filter_function is None else filter_function
    filter_words = set() if filter_function is None else filter_words
    
    # tokenize
    tokens = tokenize(sentence)
    tokens = [t for t in tokens if t not in filter_words and filter_function(t)]

    return " ".join(tokens)

def lemmatize(sentence: Union[str, List[str]]) -> List[str]:
    lem = WordNetLemmatizer()
    tokens = tokenize(sentence) if isinstance(sentence, str) else sentence
    # lemmatize
    tokens = [lem.lemmatize(t.strip().lower()) for t in tokens]
    return tokens

# filtering functions

def to_lower(text: str) -> str:
    return text.lower()

def no_extra_spaces(text: str) -> str:
    return re.sub('\s+', ' ', text)

def no_extra_chars(text: str) -> str:
    return re.sub(r"[^a-zA-Z\s,!.;:'-]+", ' ', text).strip() 

def sub_regex(text: str, regex: str, sub: str = '') -> str:
    return re.sub(regex, sub, text)

# the reduce the number of UNK tokens, we will convert certain tokens with their Name Entity.
# the spacy packages is perfect for our purposes


SPACY_LABELS = {"NORP": "group", "ORG": "organization", "GPE": "location", "LOC": "location", "WORK_OF_ART": "art", "FAC": "facility"}

def _uniform_ne(doc_obj, label_name_map: Dict):
    last_ne = None
    tokens = []
    for t in doc_obj:
        if t.ent_type != 0:
            if last_ne is None or last_ne != t.ent_type:
                # append label in this case, make sure the useful name is passed
                tokens.append((label_name_map[t.ent_type_] if (t.ent_type_ in label_name_map) else t.ent_type_))
                # tokens.append(t.ent_type_)
        else:
            tokens.append(t.text)
        last_ne = t.ent_type
        
    return " ".join(tokens)        

def uniform_ne_batched(strings: Iterable[str], nlp = None) -> List[str]: 
    # set the default nlp objects
    nlp = spacy.load("en_core_web_sm") if nlp is None else nlp
    # create the pipeline: only keeps components relevant to NER
    pipe = nlp.pipe(texts=strings, disable=['tagger', 'attribute_ruler', 'senter'], )
    
    return [_uniform_ne(doc, SPACY_LABELS) for doc in pipe]
