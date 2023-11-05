# Practical ML & DL
## Assignment 1:  Report 1
## By Ayhem Bouabid

## 1) Task Description:
Text Detoxification Task is a process of transforming the text with toxic style into the text with the same meaning but with neutral style.

> Formal definition of text detoxification task can be found in [Text Detoxification using Large Pre-trained Neural Models by Dale et al., page 14](https://arxiv.org/abs/2109.08914)

Your assignment is to create a solution for detoxing text with high level of toxicity. It can be a model or set of models, or any algorithm that would work. 


## Approaches
Going through several academic papers, we can see that Text detoxification falls under the umbrealla of a slightly more general task: ***Style Transfer***: 

<p align="center">
The task of converting input text to a target text with different attributes / style.
</p>


My humble review of the academic literature led to the following findings: 

1. Generative Adversarial networks, Sequence to Sequence training and conditional Text generation are the main approaches, Point-wise models

2. Unlike their tramendous success in Computer Vision, GANs seem to fell short in this task as they produce low-quality (unnatural) sentences (grammatically), according to the recommended paper and [this paper](https://aclanthology.org/N18-1169.pdf)

3. Point-wise approaches are systems that focus on preserving as much as possible of the original input while changing the markers of the style (the parts of the text that reflect the input's style)


4. Even though the Point-wise approaches outperform Gans, we can see that they do not have as much potential as Text generation models and sequence to sequence models as Generative models capture the hidden relations between style and the entirety of a sentence that might not be necessarily expressed in specific parts of a sentence.

### Baseline

With these observations in mind, I planned my first solution: 

1. Based on [Delete, Retrieve, Generate:](https://aclanthology.org/N18-1169.pdf), I associated a toxicity score for each uni and bi-gram as follows:
    
    * For each unigram, I compute the following expression: 
    $$ \frac{N_d(toxic, x) + \lambda}{N_d(neutral, x) + \lambda}$$ 

    where $x$ is a uni-gram or a bi-gram and $N_d(class, x)$ is the number of documents of class 'label' where 'x' is present. The score is calculated using the combination of 2 datasets

2. I calculate the toxicity threshold: $threshold_{uni}$ and $threshold_bi$. They are chosen as the $40\%$ percentile of all toxicity scores. If 'x' has a toxicity score higher than the threshold then, it is considered a toxicity marker

3. Given a sentence 's', process it, tokenize it, collect all unigrams, bi grams. If any uni-bi gram is detected as toxicity marker, mask it.

4. The sentence is now masked from step '3'. Keeping in mind that this is just a baseline, I pass the resulting sentece through a pretrained masked LM. The output of the latter is the result of this solution.

[Delete, Retrieve, Generate:](https://aclanthology.org/N18-1169.pdf) suggested several interesting ideas: 

1. Simply Delete the toxicity markers and use a template-based generation: unlikely to be grammatically correct

2. In the given dataset, Find a neural (non-toxic) sentence with the closest meaning to the original text and return it.


Both of these ideas are interesting but definitely not scalable. I came to the conclusion of using a Masked Language Model since it  will be capable of capturing the general meaning of the sentence (if toxicity attributes are major parts of the sentences), decrease toxicity and keep the grammatical structure intact.

## Summarization as a detoxification mechanism

My next step had to be more sophisticated, well many things are more sophisticated than a IDF-like estimation of the toxicity of a word. Thankfully, the [paper](https://aclanthology.org/P19-1427/) mentioned in the ***RELATED WORK*** got my attention. 

What about about simply laveraging a sequence to sequence model to learn the mapping between ***toxic*** and ***neutral*** comments ? 

I have many choices, so I resorted to my data to get more ideas.

Some simple data exploration suggested that the number of stop words in toxic comments tend to be larger than in the neutral ones. I developped a simple hypothesis: 

<p align="center">
Concise sentences tend to be less toxic.
</p>

Thus, My 2nd solution was to train a Sequence to Sequence summarizer on the dataset.


## Additional Notes

* The processing involved the following operations: 

1. simple non-alpha characters removal, stripping, simple regex rules

2. Using Spacy Entity Recongition model to replace entities with given default words (decreasing the size of the vocabulary and the uni-bi grams computations)

3. The summarizer was training on a portion of the dataset where the targets were shorted than the sources.


## Results
