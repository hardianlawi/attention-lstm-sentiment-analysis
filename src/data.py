import tensorflow_datasets as tfds


def get_data():

    train_data, validation_data, test_data = tfds.load(
        name="imdb_reviews",
        split=("train[:95%]", "train[95%:]", "test"),
        as_supervised=True,
    )


    def _convert2numpy(data):
        sentences = []
        labels = []
        for x, y in tfds.as_numpy(data):
            sentences.append(x.decode("utf-8"))
            labels.append(int(y))
        return sentences, np.array(labels)


    str_X_train, y_train = _convert2numpy(train_data)
    str_X_val, y_val = _convert2numpy(validation_data)
    str_X_test, y_test = _convert2numpy(test_data)

    return (str_X_train, y_train), (str_X_val, y_val), (str_X_test, y_test)
