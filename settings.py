import os
import pickle

# Process text config
ALLOWED_KEYWORDS = {
    'to_lower',
    'remove_space',
    'remove_special',
    'remove_number',
}

# Data path config
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TUOITRE_DATA_FOLDER = os.path.join(BASE_DIR, 'TUOITRE')
FULL_DATA_FOLDER = os.path.join(TUOITRE_DATA_FOLDER, 'text')
SUMMARY_DATA_FOLDER = os.path.join(TUOITRE_DATA_FOLDER, 'summary')

# Doc2vec config
TRAINED_MODEL_FOLDER = os.path.join(BASE_DIR, 'trained_model')
DOC2VEC_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'doc2vec.model')
DOC2VEC_MAX_EPOCHS = 50
DOC2VEC_VECTOR_SIZE = 2048
DOC2VEC_ALPHA = 0.001
DOC2VEC_MIN_ALPHA = 0.00025
DOC2VEC_MIN_COUNT = 1
DOC2VEC_DM = 1  # distributed memory

# TF-IDF config
TFIDF_MAX_DF = 1.
TFIDF_MIN_DF = 1
TFIDF_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'tf_idf.pkl')
TFIDF_MAX_FEATURE = 2048

# SVD config
SVD_OUTPUT_FEATURE = 256
SVD_SOLVER = 'randomized'
SVD_ITERATION = 10
SVD_DOC2VEC_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'svd_doc2vec_model.pkl')
SVD_TFIDF_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'svd_tfidf_model.pkl')

# KNN config
KNN_N_NEIGHBORS = 10
KNN_DOC2VEC_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'knn_doc2vec_model.pkl')
KNN_TFIDF_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'knn_tdidf_model.pkl')


FILE_LIST_PATH = os.path.join(BASE_DIR, 'file_list.pkl')
with open(FILE_LIST_PATH, 'rb') as f:
    file_list = pickle.load(f)
    file_list = [os.path.join(FULL_DATA_FOLDER, f) for f in file_list]

# Stop word
STOP_WORD_FILE = os.path.join(BASE_DIR, 'stop_words.txt')
stop_words = []
with open(STOP_WORD_FILE, 'r') as f:
    lines = f.readlines()
    for l in lines:
        stop_words.append(l.strip())
