Ayhem Bouabid

# Final

Even though the summarizer preserved the meaning reasonably well, it is could not tackle the toxicity issue. The next step was to incorporate the loss in the training of the Seq2Seq model. Easier said than done ?

## Issues 
To effectively incorporate the toxicity, a mathematical loss should be introduced to the backward pass. 

Toxicity estimation is a crucial part of the evaluation process suggested by the authors of the paper. Such estimation is computed by a Roberta-based classification model. Referring to hugging face sequence to sequence page, we can see that Roberta is not designed for Seq2Seq tasks. 

Therefore, the output of any other seq2seq module is unlikely to be tokenized by the toxicity classifier (raising errors), or even worse, accepting the input but with a totally different semantics. 

## My solution
Well, the answer seemed quite straightforward. What if I create a model that accepts the output of the seq2seq decoder and ***mimics*** the original classifier output (or at least the best approximation we can have). Well This procedure would be very similar to ***[KNOWLEDGE DISTILLATION](https://arxiv.org/pdf/1503.02531.pdf)***

### KD the author's toxic classifier
I used the same approach introduced by [KNOWLEDGE DISTILLATION](https://arxiv.org/pdf/1503.02531.pdf). I trained a BartClassification Model to mimic the outputs of the toxic classifier by minimizing the Cross entropy between the outputs of the 2 models (after applying softmax).


### The final seq2seq
The Bart model was simply fine-tuned to minimize the following loss:

$$ L = L_1 + 0.1 * L_2$$

where 
* $L_1$ is the sequence 2 sequence loss (aggregated Cross entropy between predicted and ) 
* $L_2$ is the Cross entropy of the outputs of the 2 models.

* NOTE: My toxicity classifier was trained on the [Jigsaw Toxic comments Dataset](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge): The same dataset on which the original model was trained.

* Note: The custom classifier's was forzen during the training of the seq2seq model.

## Rewriting toxic comments
The Bart model was fine-tuned on the test split of (a part) ParMNT dataset with the following parameters:

- Batch size: $35$
- Learning rate: $5\cdot10^{-5}$
- Optimizer: AdamW

I used the same evaluation process as the authors (according to their github repository)

* the BLEU score, which measures the similarity between the generated and the reference sentences, 

* the SIM score, which measures the semantic similarity between the generated and the reference sentences using sentence embeddings, 

* the ACC score, which measures the accuracy of style transfer between the generated sentences, the 

* FL score, which measures the fluency of the generated sentences, 

* the J score [<a href="#user-content-3">3</a>] which is a combination of last three metrics.

The model was evaluated on the validation part of ParaMNT. The results are presented in the table below:
<div align='center' style='min-width: 75%; font-size: 1.1em'>

| Metric | Value |
| --- | --- |
| ACC | 0.7749 |
| SIM | 0.7229 |
| FL | 0.8119 |
| J | 0.45 |
| BLEU | 0.5027 |

</div>
