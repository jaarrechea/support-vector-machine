import os
import sys
import argparse 

import cPickle as pickle 
import numpy as np
from sklearn.multiclass import OneVsOneClassifier
from sklearn.svm import LinearSVC
from sklearn import preprocessing

def parsear_argumentos():
    parser = argparse.ArgumentParser(description='Entrena los modelos clasificadores')
    parser.add_argument("--feature-map-file", dest="feature_map_file", required=True,
            help="Archivo de entrada pickle que contenien el mapa de características")
    parser.add_argument("--svm-file", dest="svm_file", required=False,
            help="Archivo de salida donde el modelo SVM se va a almacenar")
    return parser

class EntrenadorDelClasificador(object):
    def __init__(self, X, label_words):
        self.le = preprocessing.LabelEncoder()  
        self.clf = OneVsOneClassifier(LinearSVC(random_state=0))

        y = self._encodeLabels(label_words)
        X = np.asarray(X)
        self.clf.fit(X, y)

    def _fit(self, X):
        X = np.asarray(X)
        return self.clf.predict(X)
        
    def _encodeLabels(self, labels_words):
        self.le.fit(labels_words) 
        return np.array(self.le.transform(labels_words), dtype=np.float32)

    def classify(self, X):
        labels_nums = self._fit(X)
        labels_words = self.le.inverse_transform([int(x) for x in labels_nums]) 
        return labels_words

if __name__=='__main__':
    args = parsear_argumentos().parse_args()
    feature_map_file = args.feature_map_file
    svm_file = args.svm_file

    # Cargar el mapa de características
    with open(feature_map_file, 'r') as f:
        feature_map = pickle.load(f)

    # Extraer el vector de características y las etiquetas
    labels_words = [x['label'] for x in feature_map]
    dim_size = feature_map[0]['feature_vector'].shape[1]  
    X = [np.reshape(x['feature_vector'], (dim_size,)) for x in feature_map]
    
    # Entrenamiento del SVM
    svm = EntrenadorDelClasificador(X, labels_words) 
    if args.svm_file:
        with open(args.svm_file, 'w') as f:
            pickle.dump(svm, f)
