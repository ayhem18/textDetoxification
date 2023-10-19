"""
This is a utility script
"""


# preprocessing functions
import re
import requests
import os

from typing import List, Set

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

def simple_stop_words() -> Set:
    # create a set object out of the list of stop words provided by the nltk package
    pass



def to_lower(text: str) -> str:
    return text.lower()

def no_extra_spaces(text: str) -> str:
    return re.sub('\s+', ' ', text)

def no_extra_chars(text: str) -> str:
    return re.sub(r'[^a-zA-Z\s,!.;:-]+', ' ', text) 

def sub_regex(text: str, regex: str, sub: str = '') -> str:
    return re.sub(regex, sub, text)
