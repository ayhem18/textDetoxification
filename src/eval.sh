#!/bin/bash

if [! -d "source_nlp" ]; then
    git clone https://github.com/s-nlp/detox source_nlp
fi


cd source_nlp
pip install -r requirements.txt
cd emnlp2021
bash prepare.sh

python -m metric/metric.py -i $1 -p $2
# bash src/eval.sh MY_REF MY_PREDS