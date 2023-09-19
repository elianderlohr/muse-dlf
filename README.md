# Master Thesis

Playground and potential code for my master thesis.

__Rough Idea__:

1. Take FrameAxis Bias and Intensity for each microframe (positive and negative pool of words) for each sentence
2. Combine with FRISS output for each sentence
3. Combine with Sentence Embedding
4. Train a classifier to predict the Document Frame from The Media Frames Corpus (Card et al., 2015)

## Data

### The Media Frames Corpus

Consist out of 14 frames:
1. Economic
2. Capacity and Resources
3. Morality
4. Fairness and Equality
5. Legality, Constitutionality, Jurisdiction
6. Policy Prescription and Evaluation
7. Crime and Punishment
8. Security and defense
9. Health and Safety
10. Quality of Life
11. Cultural Identity
12. Public opinion
13. Political
14. External Regulation and Reputation

    [The Media Frames Corpus: Annotations of Frames Across Issues](https://aclanthology.org/P15-2072) (Card et al., ACL-IJCNLP 2015)

### Dataset

Use of SemEval-2023 Task 3 dataset. The dataset consists out of 3 subtasks:
- Subtask 1: Classify news articles as opinion pieces, objective news reporting, or satire.
- Subtask 2: Identify frames used in news articles.
- Subtask 3: Identify persuasion techniques in each paragraph of news articles.

Available languages are English, Spanish, French, German, Greek, Italian, .. and Russian.

> Some more Infos about the dataset can be seen under [notebooks\data-visualization.ipynb](notebooks\data-visualization.ipynb).

#### Subtask 1:

Classifies if a article is either an opinion piece, objective news reporting, or satire.

#### Subtask 2:

Identifies frames used in news articles. The frames are based on the Media Frames Corpus (Card et al., 2015).

A article can have multiple frames. The frames are divided into 14 categories (see above).

#### Subtask 3:

Identifies persuasion techniques in each paragraph of news articles.

E.g. `Loaded_Language, Conversation_Killer, ...`

The Infos are available not only per sentence but also per span. So we know which persuasion technique is used in which span.

Example:

![Example](/assets/imgs/subtask3_example.png)


    [SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup](https://aclanthology.org/2023.semeval-1.317) (Piskorski et al., SemEval 2023)

## TODO

- [x] Use BERT for Sentence Embedding
- [ ] Implement and Play with FrameAxis
- [ ] Implement and Play with FRISS
- [ ] Identify what types of microframes are used for FrameAxis
    - As Card et al. (2015) only uses frames which are rather "neutral" and not consist out of positive/negative words. 
    - Or try to split frame into sub group of positive/negative words
- [ ] Use newly defined microframes for FrameAxis
- [ ] Combine FrameAxis bias/intensity with FRISS output and sentence embedding
- [ ] Train a classifier to predict the Document Frame


__IMPORTANT QUESTION:__

- Do I want to predict the Document Frame, Frame for each Sentence or also the Leaning of the frame. 

    1. Document Frame: Predict the frame of the whole document
    2. Frame for each Sentence: Predict the frame for each sentence
    3. Leaning of the frame: Predict the leaning of the frame (e.g. positive/negative)
        - E.g. is the reader of the article persuaded to a positive or negative view of the frame 


## Literature

Frames:

    [The Media Frames Corpus: Annotations of Frames Across Issues](https://aclanthology.org/P15-2072) (Card et al., ACL-IJCNLP 2015)

SemEval:

    [SemEval-2023 Task 3: Detecting the Category, the Framing, and the Persuasion Techniques in Online News in a Multi-lingual Setup](https://aclanthology.org/2023.semeval-1.317) (Piskorski et al., SemEval 2023)

FrameAxis:

    [FrameAxis: Characterizing Microframe Bias and Intensity with Word Embedding](https://arxiv.org/abs/2002.08608) (Kwak et al. 2020)

FRISS:

    [Framing Unpacked: A Semi-Supervised Interpretable Multi-View Model of Media Frames](https://aclanthology.org/2021.naacl-main.174) (Khanehzar et al., NAACL 2021)