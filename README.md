# Word-level translator using LSTMs

This notebook shows how to train with Amazon SageMaker an NLP model that translates from English to a series of languages. The list of languages is limited by the available training data.

The model uses an LSTM based encoder / decoder architeecture, inspired by https://github.com/hlamba28/Word-Level-Eng-Mar-NMT.

The training time depends on the dataset available. Typically an epoch is trained in 30 minutes for small datasets (for example, Hungarian) and 1.5 - 2 hours for big datasets (for example, Italian). With 15-20 epochs you can already get an acceptable result.

The dataset was created from a curated list of anki flash cards (https://ankiweb.net/shared/decks/). Each translated sentences is annotated with CC license string.
