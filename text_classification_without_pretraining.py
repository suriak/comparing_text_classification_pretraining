import os
import re
import sys
from time import time
from datetime import timedelta
import numpy as np
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils
from keras.layers import Dense, Input, Reshape, Concatenate, Flatten
from keras.layers import Conv2D, MaxPool2D, Embedding, Dropout
from keras.models import Model
from keras.optimizers import Adam
from keras.utils import plot_model

from sklearn.preprocessing import LabelEncoder


class SentenceClassifier:
    def __init__(self):
        self.DATA_DIR = './data-set/trec/'
        self.MAX_SEQUENCE_LENGTH = 51
        self.EMBEDDING_DIM = 100
        self.LABEL_COUNT = 0
        self.WORD_INDEX = dict()
        self.LABEL_ENCODER = None
        self.VALIDATION_SPLIT = 0.2

    def clean_str(self, string):
        """
        Cleans each string and convert to lower case. Copied from author's implementation.
        :param string: Each sentence from data set.
        :return: Processed sentence in lower case.
        """
        string = re.sub(r"[^A-Za-z0-9:(),!?\'\`]", " ", string)
        string = re.sub(r" : ", ":", string)
        string = re.sub(r"\'s", " \'s", string)
        string = re.sub(r"\'ve", " \'ve", string)
        string = re.sub(r"n\'t", " n\'t", string)
        string = re.sub(r"\'re", " \'re", string)
        string = re.sub(r"\'d", " \'d", string)
        string = re.sub(r"\'ll", " \'ll", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
        return string.strip().lower()

    def load_data_train(self):
        """
        Reads the training data set from data directory and splits into sentences (X) and labels (Y).
        :return: Two list of lists for sentences (X_Train) and labels (Y_Train)
        """
        data_set = list(open(os.path.join(self.DATA_DIR, 'train_5500.label'), encoding='utf-8', errors='replace').readlines())
        data_set_cleaned = [self.clean_str(sent) for sent in data_set]
        Y_Train = [s.split(' ')[0].split(':')[0] for s in data_set_cleaned]
        X_Train = [s.split(" ")[1:] for s in data_set_cleaned]
        return X_Train, Y_Train

    def load_data_test(self):
        """
        Reads the test data set from data directory and splits into sentences (X) and labels (Y).
        :return: Two list of lists for sentences (X_Test) and labels (Y_Test)
        """
        data_set = list(open(self.DATA_DIR + 'TREC_10.label', encoding='utf-8', errors='replace').readlines())
        data_set_cleaned = [self.clean_str(sent) for sent in data_set]
        Y_Test = [s.split(' ')[0].split(':')[0] for s in data_set_cleaned]
        X_Test = [s.split(" ")[1:] for s in data_set_cleaned]
        return X_Test, Y_Test

    def integer_encode_train(self, X_Train, Y_Train):
        """
        Encodes each sentence into integer representing and pad to same length. For labels, each category is mapped to
        an integer using label encoder and then converts to one-hot representation.
        :param X_Train: Sentences from training data set.
        :param Y_Train: Labels for each sentences in training data set.
        :return: Integer encoded sentences padded to same length (X_train_encoded_padded) and labels in one-hot
        representation (Y_train_one_hot).
        """
        tokenizer = Tokenizer(lower=True, char_level=False)
        tokenizer.fit_on_texts(X_Train)
        self.WORD_INDEX = tokenizer.word_index
        X_train_encoded = tokenizer.texts_to_sequences(X_Train)
        X_train_encoded_padded = pad_sequences(X_train_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        encoder = LabelEncoder()
        encoder.fit(Y_Train)
        self.LABEL_ENCODER = encoder
        Y_train_encoded = encoder.transform(Y_Train)
        Y_train_one_hot = np_utils.to_categorical(Y_train_encoded)
        self.LABEL_COUNT = Y_train_one_hot.shape[1]
        print("\tUnique Tokens in Training Data: %s" % len(self.WORD_INDEX))
        print("\tShape of data tensor (X_train): %s" % str(X_train_encoded_padded.shape))
        print("\tShape of label tensor (Y): %s" % str(Y_train_one_hot.shape))
        return X_train_encoded_padded, Y_train_one_hot

    def integer_encode_test(self, X_Test, Y_Test):
        """
        Encodes each sentence in test data set into integer representation using the same WORD_INDEX used for training
        data set and pad to same length as of training data. Labels are converted similarly to its corresponding one-hot
        representation using the LABEL_ENCODER used for training data set.
        :param X_Test: Sentences from test data set.
        :param Y_Test: Labels for each sentence in test data set.
        :return: Integer encoded sentences padded to same length (X_test_encoded_padded) and labels in one-hot
        representation (Y_test_one_hot).
        """
        X_test_encoded = list()
        for sentence in X_Test:
            x_test = [self.WORD_INDEX[w] for w in sentence if w in self.WORD_INDEX]
            X_test_encoded.append(x_test)
        X_test_encoded_padded = pad_sequences(X_test_encoded, maxlen=self.MAX_SEQUENCE_LENGTH, padding='post')

        Y_test_encoded = self.LABEL_ENCODER.transform(Y_Test)
        Y_test_one_hot = np_utils.to_categorical(Y_test_encoded)
        print("\tUnique Tokens in Test Data (this should be same as in Training Data): %s" % len(self.WORD_INDEX))
        print("\tShape of data tensor (X_test): %s" % str(X_test_encoded_padded.shape))
        print("\tShape of label tensor (Y_test): %s" % str(Y_test_one_hot.shape))
        return X_test_encoded_padded, Y_test_one_hot

    def train_dev_split(self, X_train_encoded_padded, Y_train_one_hot):
        """
        Splits the training data set into training and validation(dev) set based on the value specified by
        the parameter VALIDATION_SPLIT.
        :param X_train_encoded_padded: Training sentences which are integer encoded and padded.
        :param Y_train_one_hot: Labels for training sentences in one-hot format.
        :return: Training sentences (x_train) and corresponding labels(y_train). Validation sentences (x_val) and
        corresponding labels (y_val).
        """
        indices = np.arange(X_train_encoded_padded.shape[0])
        np.random.shuffle(indices)
        X_train_encoded_padded = X_train_encoded_padded[indices]
        Y_train_one_hot = Y_train_one_hot[indices]
        num_validation_samples = int(self.VALIDATION_SPLIT * X_train_encoded_padded.shape[0])
        x_train = X_train_encoded_padded[:-num_validation_samples]
        y_train = Y_train_one_hot[:-num_validation_samples]
        x_val = X_train_encoded_padded[-num_validation_samples:]
        y_val = Y_train_one_hot[-num_validation_samples:]
        return x_train, y_train, x_val, y_val

    def sentence_classifier_cnn(self, x_train, y_train):
        """
        Creates a model as specified by Kim in the famous paper Convolutional Neural Networks for Sentence
        Classification [2014]. Few hyper-parameters are changed from the original paper in this implementation. This
        version is the static model of CNN mentioned in the publication.
        :param embedding_matrix: Pre-trained vectors are loaded to embedding layer and is made static by the parameter
        trainable=False.
        :return: Model is returned. The pictorial representation is saved to working folder as a png file.
        """
        inputs = Input(shape=(self.MAX_SEQUENCE_LENGTH,), dtype='int32')
        X_input = Embedding(input_dim=(len(self.WORD_INDEX) + 1), output_dim=self.EMBEDDING_DIM,
                            input_length=self.MAX_SEQUENCE_LENGTH, trainable=True)(inputs)

        X_input = Reshape((self.MAX_SEQUENCE_LENGTH, self.EMBEDDING_DIM, 1))(X_input)
        # print(X_input)

        X1 = Conv2D(128, kernel_size=(3, self.EMBEDDING_DIM), padding='valid', kernel_initializer='normal',
                    activation='relu',
                    name='conv1Filter1')(X_input)
        maxpool_1 = MaxPool2D(pool_size=(48, 1), strides=(1, 1), padding='valid', name='maxpool1')(X1)

        X2 = Conv2D(128, kernel_size=(4, self.EMBEDDING_DIM), padding='valid', kernel_initializer='normal',
                    activation='relu',
                    name='conv1Filter2')(X_input)
        maxpool_2 = MaxPool2D(pool_size=(47, 1), strides=(1, 1), padding='valid', name='maxpool2')(X2)

        X3 = Conv2D(128, kernel_size=(5, self.EMBEDDING_DIM), padding='valid', kernel_initializer='normal',
                    activation='relu',
                    name='conv1Filter3')(X_input)
        maxpool_3 = MaxPool2D(pool_size=(46, 1), strides=(1, 1), padding='valid', name='maxpool3')(X3)

        concatenated_tensor = Concatenate(axis=1)([maxpool_1, maxpool_2, maxpool_3])

        flatten = Flatten()(concatenated_tensor)
        dropout = Dropout(0.5)(flatten)
        output = Dense(units=self.LABEL_COUNT, activation='softmax', name='fully_connected_affine_layer')(dropout)

        model = Model(inputs=inputs, outputs=output, name='intent_classifier')
        print("Model Summary")
        print(model.summary())
        adam = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)
        model.compile(loss='categorical_crossentropy',
                      optimizer=adam,
                      metrics=['accuracy'])
        model.fit(x_train, y_train,
                  batch_size=64,
                  epochs=25,
                  verbose=2)
        plot_model(model, to_file='text_classifier_cnn.png')
        return model


if __name__ == '__main__':
    sampleClassifier = SentenceClassifier()
    print("Loading Training Data set...")
    X_Train, Y_Train = sampleClassifier.load_data_train()
    print("Encoding Training Data set...")
    X_train_encoded_padded, Y_train_one_hot = sampleClassifier.integer_encode_train(X_Train, Y_Train)
    # print("Splitting Data set to Train and Validation set...")
    # x_train, y_train, x_val, y_val = sampleClassifier.train_dev_split(X_train_encoded_padded, Y_train_one_hot)
    train_start_time = time()
    print("Training Started...")
    model = sampleClassifier.sentence_classifier_cnn(X_train_encoded_padded, Y_train_one_hot)
    train_end_time = time()
    print("Training took %s units." % (str(timedelta(seconds=train_end_time - train_start_time))))

    # Evaluating the model
    print("Evaluating the model...")
    print("Loading Test Data set...")
    X_Test, Y_Test = sampleClassifier.load_data_test()
    print("Encoding Test Data set...")
    X_test_encoded_padded, Y_test_one_hot = sampleClassifier.integer_encode_test(X_Test, Y_Test)
    print("Evaluating the model on the Test Data set...")
    scores = model.evaluate(X_test_encoded_padded, Y_test_one_hot, verbose=0)
    print("%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
