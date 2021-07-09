import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.python.keras.optimizer_v2.gradient_descent import SGD

lemmatizer = WordNetLemmatizer()

import json
import pickle

import numpy as np
from keras.models import Sequential,load_model
from keras.layers import Dense, Dropout
import random
import matplotlib.pyplot as plt

try:
    model = load_model('data/model.h5')
    intents = json.loads(open('data/intents.json').read())
    words = pickle.load(open('data/words.pkl', 'rb'))
    classes = pickle.load(open('data/classes.pkl', 'rb'))
    history = pickle.load(open("data/trainHistory.pkl", 'rb'))

except:
    words = []
    classes = []
    documents = []
    ignore_words = ['?', '!','*','(',')','&']
    data_file = open('data/intents.json').read()
    intents = json.loads(data_file)

    for intent in intents['intents']:
        for pattern in intent['patterns']:
            w = nltk.word_tokenize(pattern)
            words.extend(w)
            documents.append((w, intent['tag']))
            if intent['tag'] not in classes:
                classes.append(intent['tag'])

    words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_words]
    words = sorted(list(set(words)))

    classes = sorted(list(set(classes)))

    pickle.dump(words, open('data/words.pkl', 'wb'))
    pickle.dump(classes, open('data/classes.pkl', 'wb'))

    training = []

    output_empty = [0] * len(classes)

    for doc in documents:

        bag = []

        pattern_words = doc[0]

        pattern_words = [lemmatizer.lemmatize(word.lower()) for word in pattern_words]

        for w in words:
            bag.append(1) if w in pattern_words else bag.append(0)

        output_row = list(output_empty)
        output_row[classes.index(doc[1])] = 1

        training.append([bag,output_row])

    random.shuffle(training)
    training = np.array(training,dtype="object")
    train_x = list(training[:, 0])
    train_y = list(training[:, 1])

    model = Sequential()
    model.add(Dense(128, input_shape=(len(train_x[0]),), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(train_y[0]), activation='softmax'))

    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])


    print(model.summary())


    history = model.fit(np.array(train_x),np.array(train_y),epochs=200,batch_size=5,verbose=1)
    model.save('data/model.h5', history)

    with open('data/trainHistory.pkl', 'wb') as hist:
        pickle.dump(history.history, hist)

    model = load_model('data/model.h5')
    intents = json.loads(open('data/intents.json').read())
    history = pickle.load(open("data/trainHistory.pkl", 'rb'))
    words = pickle.load(open('data/words.pkl', 'rb'))
    classes = pickle.load(open('data/classes.pkl', 'rb'))


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def bow(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                bag[i] = 1
    return(np.array(bag))

def predict_class(sentence, model):

    p = bow(sentence,words)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": r[1]})
    return return_list

def getResponse(ints, intents_json):

    global result
    try:
        probabiltiy = ints[0]["probability"]
        if (probabiltiy > 0.7038719):
            tag = ints[0]['intent']
            list_of_intents = intents_json['intents']
            for i in list_of_intents:
                if (i['tag'] == tag):
                    result = random.choice(i['responses'])
                    break
        else:
            result = "Hey..I didn't get you!!Try asking once again!"
    except:
        result = "Hey..I didn't get you!!Try asking once again!"
    return result

def plot():
    plt.title('Loss')
    plt.plot(history['loss'], label='training loss',color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    plt.title('Accuracy')
    plt.plot(history['accuracy'], label='training accuracy',color='orange')
    plt.xlabel('Epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.show()

plot()

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res
