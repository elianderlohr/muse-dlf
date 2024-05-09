# TODO

- [x] Test if balancing of unsupervised loss (e.g. divide by num sentences) is necessary/useful
  - Balacing by number of batch_size x num_sentences x num_args is necessary and useful
- [ ] Test if adding frameaxis data to the supervised sentenced prediction is useful

## TODO FrameAxis

- [x] Apply lemmatization to find words for microframe creation
- [ ] Use the span annotations to see how the microframes are predicting

## TODO SRL / MuSE-DLF

- [x] Sentences are sometimes not correctly split e.g. "Our Sen. John McCain" -> "Our Sen." and "John McCain" which leads to wrong SRL results
