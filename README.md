# Sentiment Analysis using Attention and LSTM

## Dataset

The dataset used is the famous IMDB dataset which is downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). For the sake of convenience, the dataset is uploaded here and could be found in the `data` directory.

## Using Docker




## Preparing Environment

Firstly, make sure that `conda` is available in the environment. Then, run the command below to install the environment required to run the codes:

```bash
make env

# To turn on the environment
conda activate attention-lstm-sentiment-analysis
```

## Train

After running `make env`, to train the model, simply run the command below:

```bash
# lstm
make train LOG_DIR=models/lstm MODEL_TYPE=lstm

# attention+LSTM
make train LOG_DIR=models/attention MODEL_TYPE=attention
```

**Note**: The parameters could be exposed in the `Makefile` or set using config file. However, as this is not meant to be production-ready codes, running `make train` will use all the default configs set.

## Running the webservice

The app could be built and run by running the command below:

```bash
# lstm
make app LOG_DIR=models/lstm MODEL_TYPE=lstm

# Attention
make app LOG_DIR=models/attention MODEL_TYPE=attention
```

## Using Docker

All the steps above could be run immediately as long as you have docker installed on your local. You can simply run the command below:

```bash
# LSTM
make build_app MODEL_TYPE=lstm
make run_app MODEL_TYPE=lstm

# Attention
make build_app MODEL_TYPE=attention
make run_app MODEL_TYPE=attention
```

## Answers

### What is attention mechanism? Why can it improve LSTM?

> Attention mechanism was developed based on humans' way of perceiving information (images, texts, etc) by focusing more on certain parts depending on the objective. Internally, it is learning a set of importance weights of different parts of, in this case, sentence that tells us which words of the sentence are more important when determining the sentiment of the sentence. The biggest advantage of attention over LSTM is that it helps the gradients to backpropagate to each timestep because it creates a direct connection from each timestep to the output layer as opposed to stack of multiplications in LSTM. This is particularly important when we are dealing with long timesteps/sentences due to the vanishing/exploding gradient problem.

### Is the attention mechanism useful in this problem?

> Yes, the comparison between LSTM and LSTM+Attention can be seen in [this notebook](https://github.com/hardianlawi/attention-lstm-sentiment-analysis/blob/master/notebooks/Benchmarking.ipynb). Two sets of configs were run for both LSTM and Attention+LSTM where both LSTM are of same size, but Attention+LSTM has slightly more parameters due to the attention layer. In both cases, Attention+LSTM gives almost 1% boost in terms of accuracy with only slight increase in the number of parameters.

5. - Initialize embeddings from pre-trained models trained on big corpus. This provides better prior for the model to start with. This helps the model converges faster and improves the generalizability especially the case where the words do not occur frequently in the training data.
    - Ensembling and stacking of several models could generally provide some percentage boost to the prediction power although usually with the cost of reduction in speed.
    - Bidirectional LSTM incorporates both past and future information to generate prediction. This could generally help when how you perceive your past information is influenced by your future information. The drawback of using this is you would always have to wait for the complete sentence before generating a prediction.
    - Remove LSTM altogether and use more powerful attention based algorithm e.g. BERT
7. Empty strings (OOV tokens in the string are less than x%)
8. Data validation, integration tests, unit tests, performance validation check

## TODO

1. [x] Reproduce one of these sentiment analyses using your preferred deep learning framework
2. [x] Explain conceptually what attention mechanism is, and why it can improve LSTM
3. [x] Add attention mechanism to the model that you reproduced above
4. [ ] Perform validation and benchmarking analysis to asses the usefulness of attention mechanism in this case. Is it useful? If yes, no, describe why
5. [x] What other ways can we use to improve LSTM-based sentiment analysis? If you have the time, feel free to validate against the IMDB dataset
6. [ ] wrap your model in a simple REST API that accepts a string input of the text and returns the sentiment analyses result in a json (intention is to test if the candidate can structure the code properly into different functions i.e. front load the models etc.)
7. [ ] briefly describe what kind of pre-processing or checks you will do to the string input (intention is to test if the candidate can think of problems that may surface at production)
8. [ ] in code or text, describe how you will write test cases for your model training