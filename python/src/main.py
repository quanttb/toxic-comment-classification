import pandas as pd
import time
# clean function for cleaning the dataset
from comment_cleaner import clean
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, LSTM, Embedding, Dropout, GlobalMaxPool1D
from keras.models import Model
import joblib
import io

start = time.time()

train = pd.read_csv('./data/train.csv', encoding='latin-1')

print('Reading the dataset...')
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_train = train["comment_text"].apply(lambda comment: clean(comment))

max_features = 20000
tokenizer = Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)

maxlen = 100
X_t = pad_sequences(list_tokenized_train, maxlen=maxlen)
# maxlen=200 as defined earlier
inp = Input(shape=(maxlen, ))

# size of the vector space
embed_size = 128
x = Embedding(max_features, embed_size)(inp)

output_dimention = 60
x = LSTM(output_dimention, return_sequences=True,name='lstm_layer')(x)
# reduce dimention
x = GlobalMaxPool1D()(x)
# disable 10% precent of the nodes
x = Dropout(0.1)(x)
# pass output through a RELU function
x = Dense(50, activation="relu")(x)
# another 10% dropout
x = Dropout(0.1)(x)
# pass the output through a sigmoid layer, since
# we are looking for a binary (0,1) classification
x = Dense(6, activation="sigmoid")(x)

model = Model(inputs=inp, outputs=x)
# we use binary_crossentropy because of binary classification
# optimise loss by Adam optimiser
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

print('Training Model...')
start_fitting = time.time()
batch_size = 32
epochs = 2
model.fit(X_t,y, batch_size=batch_size, epochs=epochs, validation_split=0.1)
fitting_model_time = time.time()
print('Training Model took: ', fitting_model_time - start_fitting)

# Export files
print('Exporting files...')
model.save('./data/model.h5', save_format='h5')
joblib.dump(tokenizer, './data/tokenizer.joblib')
with io.open('./data/tfjs/tokenizer.json', 'w', encoding='utf-8') as f:
  f.write(tokenizer.to_json())

end = time.time()
print('TOTAL time spent', end-start)
