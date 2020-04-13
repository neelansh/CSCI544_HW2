import os
from hw2_corpus_tool import *
import pycrfsuite
import random
import sys
import time

def get_features(utterance, first_utterance, last_speaker):
    features = []
    if(last_speaker and utterance.speaker != last_speaker):
        features.append('SPEAKER_CHANGE')
        
    if(first_utterance):
        features.append('FIRST_UTTERANCE')
    
    if(not utterance.pos):
        features.append('NO_WORDS')
        return features, utterance.speaker, utterance.act_tag
    
    for token, pos in utterance.pos:
        features.append('TOKEN_'+token)
        features.append('POS_'+pos)
    
    return features, utterance.speaker, utterance.act_tag


def create_dataset(path):
    dataset = get_data(path)
    x_final = []
    y_final = []
    for conversation in dataset:
        x = []
        y = []
        first_utterance = True
        last_speaker = None
        for utterance in conversation:
            features, last_speaker, label = get_features(utterance, first_utterance, last_speaker)
            x.append(features)
            y.append(label)
            if(first_utterance):
                first_utterance = False
        x_final.append(x)
        y_final.append(y)
    return x_final, y_final

def train(x_train, y_train):
    trainer = pycrfsuite.Trainer(verbose=False)

    for (x, y) in zip(x_train, y_train):
        trainer.append(x, y)

    trainer.set_params({
        'c1': 1.0, # coefficient for L1 penalty
        'c2': 1e-3, # coefficient for L2 penalty
        'max_iterations': 50, # stop earlier
        # include transitions that are possible, but not observed
        'feature.possible_transitions': True
    })

    trainer.train('baseline_tagger.crfsuite')
    return 


def predict(x_test):
    y_pred = []
    crftagger = pycrfsuite.Tagger()
    crftagger.open('baseline_tagger.crfsuite')

    for x in x_test:
        y_pred.append(crftagger.tag(x))

    return y_pred

def calculate_accuracy(y_pred, y_test):
    correct = 0
    for c_pred, c_test in zip(y_pred, y_test):
        for l1, l2 in zip(c_pred, c_test):
            if(l1 == l2):
                correct += 1
    return correct/sum([len(x) for x in y_test])

def save_output(y_pred, output_path):
    file = open(output_path, 'wt')
    for conv in y_pred:
        for predicted_label in conv:
            file.write(predicted_label+'\n')
        file.write('\n')
            
    file.close()


if __name__ == '__main__':
    train_path = sys.argv[1].strip()
    test_path = sys.argv[2].strip()
    output_path = sys.argv[3].strip()
    x_train, y_train = create_dataset(train_path)
    x_test, y_test = create_dataset(test_path)
    train(x_train, y_train)
    y_pred = predict(x_test)
    save_output(y_pred, output_path)
#     print(calculate_accuracy(y_pred, y_test))
    