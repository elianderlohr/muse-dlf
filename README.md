# Master Thesis on Advanced Frame Detection in Media Narratives

Master thesis on advanced frame detection in media narratives, using a novel approach combining unsupervised and supervised learning.

## Motivation

The concept of framing, deeply entrenched in psychology and sociology, significantly influences public opinion and decision-making. Framing involves presenting information in a specific manner, often through the use of narratives, symbols, or stereotypes, to subtly guide perceptions. This method is particularly evident in media, where different outlets may portray the same issue in distinct ways, leading audiences to varying conclusions. The impact of framing on public discourse and international relations is notable in events like the differing narratives presented by President Biden and President Putin regarding the Russo-Ukrainian war. Understanding and quantifying the effects of framing is crucial for a more informed and critical public discourse.

## Data Utilized

This research leverages two key datasets:

### Media Frames Corpus (MFC)

The MFC, a collection of news articles labeled with frame labels from "The Policy Frames Codebook," focuses on contentious policy issues like immigration, smoking, and gun control. This corpus includes both labeled and unlabeled articles from major U.S. newspapers spanning from 1980 to 2012, providing a rich source of data for frame analysis.

### SemEval-2023 Dataset

Comprising news articles in nine languages, the SemEval-2023 dataset covers international events from 2020 to mid-2022. It encompasses a wide range of topics, from the COVID-19 pandemic to the Russo-Ukrainian war. This dataset is annotated with frames from "The Policy Frames Codebook" and offers data in multiple languages, including English, French, German, and Russian.

## Approach

![assets\imgs\model-architecture.png](assets\imgs\model-architecture.png)

> **Note:** The approach described is not yet implemented in the codebase as it is still in development.

1. **Article Preprocessing**: Each article, composed of $N$ sentences, is divided into individual sentences to prepare for processing.

2. **Sentence Embedding**: The divided sentences are processed through a transformer-based model like Sentence-BERT (sBERT) to obtain sentence embeddings, capturing their contextual meaning.

3. **Word Embedding**: Each word in the sentences is processed through a transformer model like BERT to generate word embeddings, offering a detailed understanding of word usage.

4. **Unsupervised FRISS Module**: Operates on dictionary learning principles, deconstructing input data into a sparse representation. It identifies semantic role labels (SRLs) in sentences, extracting their embeddings. These embeddings are fed into an autoencoder, producing latent view-specific representations.

5. **Unsupervised FrameAxis Module**: Assesses each word in the sentences against predefined microframes, calculating bias measurements for each sentence. This helps understand the textual bias along specific semantic axes.

6. **Supervised Learning Module**: Combines latent SRL representations, sentence embeddings, and FrameAxis bias measurements. This module is optimized to minimize both the unsupervised loss (difference between input and reconstructed SRL embeddings) and the supervised loss (associated with predicting multiple frame labels).

7. **Frame Prediction**: The model ultimately produces predictions of the frames present in each article, resolving the multi-label problem of associating each article with multiple frames.

## Next Steps

### Experiment 1: Media Frames Corpus (MFC) Application

- **Objective**: Test FRISS-FrameAxis model's capability in identifying and interpreting media frames on MFC.
- **Methodology**: Train the integrated FRISS-FrameAxis model on MFC, focusing on frame detection efficiency.
- **Evaluation Metrics**: Compare accuracy and macro-F1 score against the original FRISS model to gauge improvements.

### Experiment 2: SemEval 2023 Dataset Extension for Multi-label Prediction

- **Objective**: Evaluate model adaptability in multi-label frame prediction using the SemEval 2023 dataset.
- **Methodology**: Retrain FRISS-FrameAxis model on SemEval 2023, adjusting for multiple frames in single texts.
- **Evaluation Metrics**: Use multi-label classification metrics like F1-score; compare results with SemEval 2023 challenge benchmarks.
