# Sentiment Analysis using Attention and LSTM

## Dataset

The dataset used is the famous IMDB dataset which is downloaded from [Kaggle](https://www.kaggle.com/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews). For the sake of convenience, the dataset is uploaded here and could be found in the `data` directory.

## Using Docker

### Build

To run the scripts using `docker`, make sure that `docker` is available on your local. Then, simply run the command below:

```bash
# LSTM
make build MODEL_TYPE=lstm

# Attention
make build MODEL_TYPE=attention
```

### API Test

After building the app, the test could be run using the command below:

```bash
# LSTM
make test MODEL_TYPE=lstm

# Attention
make test MODEL_TYPE=attention
```

*NOTE* This will need the port `8000` to be available.

### Running the App

This could be done without running the test. To run the app in the background, run the command below:

```bash
# LSTM
make run MODEL_TYPE=lstm

# Attention
make run MODEL_TYPE=attention
```

*NOTE* This will need the port `8080` to be available.

## Without Docker

### Preparing Environment

Firstly, make sure that `conda` is available in the environment. Then, run the command below to install the environment required to run the codes:

```bash
make env

# To turn on the environment
conda activate attention-lstm-sentiment-analysis
```

### Train

After running `make env`, to train the model, simply run the command below:

```bash
# lstm
make train LOG_DIR=models/lstm MODEL_TYPE=lstm

# attention+LSTM
make train LOG_DIR=models/attention MODEL_TYPE=attention
```

**Note**: The parameters could be exposed in the `Makefile` or set using config file. However, as this is not meant to be production-ready codes, running `make train` will use all the default configs set.

### Running the Webservice

The app could be built and run by running the command below:

```bash
# lstm
make app LOG_DIR=models/lstm MODEL_TYPE=lstm

# Attention
make app LOG_DIR=models/attention MODEL_TYPE=attention
```

### Testing the Webservice

```bash
# LSTM
make test_api LOG_DIR=models/lstm

# Attention+LSTM
make test_api LOG_DIR=models/attention
```

*NOTE* This will simply send the test requests generated from the `make train` step to the webservice run from `make app` step. Therefore, it is important to make sure that the test requests sent correspond to the correct model.

## Thoughts

### What is attention mechanism? Why can it improve LSTM?

> Attention mechanism was developed based on humans' way of perceiving information (images, texts, etc) by focusing more on certain parts depending on the objective. Internally, it is learning a set of importance weights of different parts of, in this case, sentence that tells us which words of the sentence are more important when determining the sentiment of the sentence. The biggest advantage of attention over LSTM is that it helps the gradients to backpropagate to each timestep because it creates a direct connection from each timestep to the output layer as opposed to stack of multiplications in LSTM. This is particularly important when we are dealing with long timesteps/sentences due to the vanishing/exploding gradient problem.

### Is the attention mechanism useful in this problem?

> Yes, the comparison between LSTM and LSTM+Attention can be seen in [this notebook](https://github.com/hardianlawi/attention-lstm-sentiment-analysis/blob/master/notebooks/Benchmarking.ipynb). Two sets of configs were run for both LSTM and Attention+LSTM where both LSTM are of same size, but Attention+LSTM has slightly more parameters due to the attention layer. In both cases, Attention+LSTM gives almost 1% boost in terms of accuracy with only slight increase in the number of parameters. Furthermore, some analysis on the Attention mechanism is done on [this notebook](https://github.com/hardianlawi/attention-lstm-sentiment-analysis/blob/master/notebooks/Attention%20%2B%20LSTM%20Analysis.ipynb). As can be seen, the model is working similarly to what has been explained [here](https://github.com/hardianlawi/attention-lstm-sentiment-analysis#what-is-attention-mechanism-why-can-it-improve-lstm).

### What other ways to improve LSTM-based sentiment analysis?

> - Initialize embeddings from pre-trained models trained on big corpus. This provides better prior for the model to start with. This helps the model converges faster and improves the generalizability especially the case where the words do not occur frequently in the training data.
> - Ensembling and stacking of several models could generally provide some percentage boost to the prediction power although usually with the cost of reduction in speed.
> - Bidirectional LSTM incorporates both past and future information to generate prediction. This could generally help when how you perceive your past information is influenced by your future information. The drawback of using this is you would always have to wait for the complete sentence before generating a prediction.
> - Remove LSTM altogether and use more powerful attention based algorithm e.g. BERT

### What kinds of inputs validation should be done to the string input?

> - Sequence length: If the sequence is too short, ideally you have to be more careful when generating prediction because it could just be too little context to tell the sentiment. In this case, if the string is empty, you could simply choose not to generate any predictions, but if the string is not empty but short, you could set some probability intervals (model is not confident in the prediction) in which the model does not generate any predictions.
> - Out of Vocabulary (OOV) percentage: Since the algorithm used doesn't support OOV words, it is important to check the percentage of the OOV tokens because if it is too high, the model would not have enough context to generate reliable prediction.
> - Language: This is essentially checking if the input language is the same as what was fit to the model.

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