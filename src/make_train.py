import click
from tensorflow.keras.datasets import imdb


@click.command()
@click.option("--vocabulary_size", type=int, default=5000)
@click.option("--batch_size", type=int, default=64)
@click.option("--epochs", type=int, default=3)
def main(vocabulary_size, batch_size, epochs):
    (X_train, y_train), (X_test, y_test) = imdb.load_data(num_words=vocabulary_size)


if __name__ == "__main__":
    main()
