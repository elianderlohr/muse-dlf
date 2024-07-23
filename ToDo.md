# TODO

- [x] Test if balancing of unsupervised loss (e.g. divide by num sentences) is necessary/useful
  - Balacing by number of batch_size x num_sentences x num_args is necessary and useful
- [x] Test if adding frameaxis data to the supervised sentenced prediction is useful

## TODO FrameAxis

- [x] Apply lemmatization to find words for microframe creation
- [x] FrameAxis Analysis:
  - [x] Analyse how only the microframe bias / intensity value for sentences annoated with span based document frame are placed in bias / intensity chart
  - [x] Show microframe bias and intensity shift per Word, e.g. identify how different words shift the bias based on the document frame
  - [x] Plot the bias and intensity per document instead of sentence

## TODO SRL / MuSE-DLF

- [x] Sentences are sometimes not correctly split e.g. "Our Sen. John McCain" -> "Our Sen." and "John McCain" which leads to wrong SRL results

# TODO Thesis

- [x] Update frameaxis preparation chapter --> with irgnoring words etc.
- [x] Add SLMuSE-DLF chapter
- [ ] Update MuSE-DLF chapter --> restructure with more info
- [ ] Explainability:
  - [ ] SRL Explainability
    - [ ] Extract dict
    - [ ] Add example sentence with annotations
  - [ ] FrameAxis Explainability
    - [ ] Generate Graph for FrameAxis
- [ ] Experiment 1:
  - [ ] Continue hyperparameter tuning chapter
  - [ ] Add results of hyperparameter tuning
  - [ ] Add results of the final model
- [ ] Experiment 2:
  - [ ] Write hyperparameter tuning chapter
  - [ ] Add results of hyperparameter tuning
  - [ ] Add results of the final model
