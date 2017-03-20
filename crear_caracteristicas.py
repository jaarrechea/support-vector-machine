import os
import sys
import argparse
import cPickle as pickle
import json

import cv2
import numpy as np
from sklearn.cluster import KMeans

def parsear_argumentos():
    parser = argparse.ArgumentParser(description='Crear características para las imágenes dadas')
    parser.add_argument("--samples", dest="cls", nargs="+", action="append", 
            required=True, help="Carpetas que contienen las imágenes de entrenamiento. \
            El primer elemento debe ser el nombre de la clase.")
    parser.add_argument("--codebook-file", dest='codebook_file', required=True,
            help="Nombre base del archivo para almacenar el codebook")
    parser.add_argument("--feature-map-file", dest='feature_map_file', required=True,
            help="Nombre base del archivo para almacenar el mapa de características")
    parser.add_argument("--scale-image", dest="scale", type=int, default=150,
            help="Tamaño hasta el que se escala cada imagen.")

    return parser

def cargar_mapas_entrada(label, input_folder):
    combined_data = []

    if not os.path.isdir(input_folder):
        print ("La carpeta " + input_folder + " no existe")
        raise IOError
        
    for root, dirs, files in os.walk(input_folder):
        for filename in (x for x in files if x.endswith('.jpg')):
            combined_data.append({'label': label, 'image': os.path.join(root, filename)})
                    
    return combined_data

class ExtractorDeCaracteristicas(object):
    def extraer_caracteristicas_imagenes(self, img):
        kps = DetectorDeDensidad().detectar(img)
        kps, fvs = ExtractorSIFT().evaluar(img, kps)
        return fvs

    def get_centroides(self, input_map, num_samples_to_fit=10):
        kps_all = []
        
        count = 0
        cur_label = ''
        for item in input_map:
            if count >= num_samples_to_fit:
                if cur_label != item['label']:
                    count = 0
                else:
                    continue

            count += 1

            if count == num_samples_to_fit:
                print ("Construir los centroides para", item['label'])

            cur_label = item['label']
            img = cv2.imread(item['image'])
            img = redimensionar_a_tamanho(img, 150)

            num_dims = 128
            fvs = self.extraer_caracteristicas_imagenes(img)
            kps_all.extend(fvs) 

        kmeans, centroids = Cuantificador().cuantificar(kps_all)
        return kmeans, centroids

    def get_vector_caracteristicas(self, img, kmeans, centroids):
        return Cuantificador().get_vector_caracteristicas(img, kmeans, centroids)

def extraer_mapa_caracteristicas(input_map, kmeans, centroids):
    feature_map = []
     
    for item in input_map:
        temp_dict = {}
        temp_dict['label'] = item['label']
    
        print "Extrayendo características de", item['image']
        img = cv2.imread(item['image'])
        img = redimensionar_a_tamanho(img, 150)

        temp_dict['feature_vector'] = ExtractorDeCaracteristicas().get_vector_caracteristicas(
                    img, kmeans, centroids)

        if temp_dict['feature_vector'] is not None:
            feature_map.append(temp_dict)

    return feature_map

class Cuantificador(object):
    def __init__(self, num_clusters=32):
        self.num_dims = 128
        self.extractor = ExtractorSIFT()
        self.num_clusters = num_clusters
        self.num_retries = 10

    def cuantificar(self, datapoints):
        kmeans = KMeans(self.num_clusters, 
                        n_init=max(self.num_retries, 1),
                        max_iter=10, tol=1.0)

        res = kmeans.fit(datapoints)
        centroids = res.cluster_centers_
        return kmeans, centroids

    def normalizar(self, input_data):
        sum_input = np.sum(input_data)
        if sum_input > 0:
            return input_data / sum_input
        else:
            return input_data

    def get_vector_caracteristicas(self, img, kmeans, centroids):
        kps = DetectorDeDensidad().detectar(img)
        kps, fvs = self.extractor.evaluar(img, kps)
        labels = kmeans.predict(fvs)
        fv = np.zeros(self.num_clusters)

        for i, item in enumerate(fvs):
            fv[labels[i]] += 1

        fv_image = np.reshape(fv, ((1, fv.shape[0])))
        return self.normalizar(fv_image)

class DetectorDeDensidad(object):
    def __init__(self, step_size=20, feature_scale=40, img_bound=20):
        self.detector = cv2.FeatureDetector_create("Dense")
        self.detector.setInt("initXyStep", step_size)
        self.detector.setInt("initFeatureScale", feature_scale)
        self.detector.setInt("initImgBound", img_bound)

    def detectar(self, img):
        return self.detector.detect(img)

class ExtractorSIFT(object):
    def evaluar(self, image, kps):
        if image is None:
            print "No es una imagen válida"
            raise TypeError

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kps, des = cv2.SIFT().compute(gray_image, kps)
        return kps, des

# Redimensionar la dimensión más pequeña a 'new_size'
# mantenimiento el ratio del aspecto
def redimensionar_a_tamanho(input_image, new_size=150):
    h, w = input_image.shape[0], input_image.shape[1]
    ds_factor = new_size / float(h)
    if w < h:
        ds_factor = new_size / float(w)
    new_size = (int(w * ds_factor), int(h * ds_factor))
    return cv2.resize(input_image, new_size) 

if __name__=='__main__':
    args = parsear_argumentos().parse_args()
    
    input_map = []
    for cls in args.cls:
        assert len(cls) >= 2, "El formato para la clase es `<etiqueta> archivo`"
        label = cls[0]
        input_map += cargar_mapas_entrada(label, cls[1])

    downsample_length = args.scale

    # Construyendo el codebook
    print "===== Construyendo el codebook ====="
    kmeans, centroids = ExtractorDeCaracteristicas().get_centroides(input_map)
    if args.codebook_file:
        with open(args.codebook_file, 'w') as f:
            pickle.dump((kmeans, centroids), f)
    
    # Datos de entrada y etiquetas
    print "===== Construyendo el mapa de características ====="
    feature_map = extraer_mapa_caracteristicas(input_map, kmeans, centroids)
    if args.feature_map_file:
        with open(args.feature_map_file, 'w') as f:
            pickle.dump(feature_map, f)

