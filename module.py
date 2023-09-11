from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay, f1_score, accuracy_score
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LSTM, Embedding
from tensorflow.keras.utils import plot_model
from tensorflow.keras.callbacks import TensorBoard
import numpy as np
import matplotlib.pyplot as plt
import re, os, datetime, pickle, json

def eda(df):
    print(df.head())
    print(df['text'][1])

def cleaned_data(text):
    for i, data in enumerate(text):
        temp = re.sub('\(.*?\)', ' ', data)
        temp = re.sub('[^a-zA-Z]', ' ', temp)
        temp = re.sub('\s[a-z]\\b', '', temp)
        text[i] = temp.lower()
    return text

def text_tokenization(text, vocab_size):
    oov_token = '<OOV>'

    tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_token)
    tokenizer.fit_on_texts(text)

    word_index = tokenizer.word_index
    print(list(word_index.items())[0:20])

    text_tokenized = tokenizer.texts_to_sequences(text)

    return text_tokenized, tokenizer

def text_pad_trunc(text_tokenized, maxlen):
    text_tokenized = pad_sequences(text_tokenized, maxlen=maxlen, padding='post', truncating='post')
    text_tokenized = np.expand_dims(text_tokenized, axis=-1)

    return text_tokenized

def model_archi(vocab_size, num_labels, embedding_dim, drop_rate, MODEL_PNG_PATH):
    model = Sequential()
    model.add(Embedding(vocab_size, embedding_dim))
    model.add(LSTM(embedding_dim, return_sequences=True))
    model.add(Dropout(drop_rate))
    model.add(LSTM(embedding_dim))
    model.add(Dropout(drop_rate))
    model.add(Dense(num_labels, activation='softmax'))

    model.summary()
    plot_model(model, to_file=MODEL_PNG_PATH, show_shapes=True)

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def model_train(model, X_train, X_test, y_train, y_test, epochs=30):
    log_dir = os.path.join(os.getcwd(), 'tensorboard_logs', datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))
    tb_callback = TensorBoard(log_dir=log_dir)

    hist = model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=epochs, callbacks=[tb_callback])

    return hist, model

def plot_history(hist):
    "Plotting training loss and training accuracy"
    plt.figure()
    plt.plot(hist.history['loss'])
    plt.plot(hist.history['val_loss'])
    plt.legend(['training loss', 'validation loss'])
    plt.show()

    plt.figure()
    plt.plot(hist.history['accuracy'])
    plt.plot(hist.history['val_accuracy'])
    plt.legend(['training accuracy', 'validation accuracy'])
    plt.show()

def model_metrics(model, ohe, X_test, y_test, category):
    "To display accuracy, f1_score, classification report, and confusion matrix"
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=1)
    y_test = np.argmax(y_test, axis=1)

    acc_scr = accuracy_score(y_test, y_pred)
    f1_scr = f1_score(y_test, y_pred, average='weighted')
    print(f"Accuracy Score: {acc_scr}, F1 Score: {f1_scr}\n\n")
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)
    print(cr)

    labels = ohe.inverse_transform(np.unique(category, axis=0))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    disp.plot(cmap=plt.cm.Blues)
    plt.show()

def model_save(MODEL_PATH, OHE_PATH, TOKEN_PATH, model, ohe, tokenizer):
    "To save model.h5, ohe.pkl, tokenizer.json"
    model.save(MODEL_PATH)
    with open(OHE_PATH, 'wb') as f:
        pickle.dump(ohe, f)

    token_json = tokenizer.to_json()
    with open(TOKEN_PATH, 'w') as f_json:
        json.dump(token_json, f_json)