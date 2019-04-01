import os

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
MAX_EPOCHS = 1
VECTOR_SIZE = 1024
ALPHA = 0.001
MIN_ALPHA = 0.00025
MIN_COUNT = 1
DM = 1

# SVD config
SVD_OUTPUT_FEATURE = 128
SVD_SOLVER = 'randomized'
SVD_ITERATION = 10
SVD_MODEL_PATH = os.path.join(TRAINED_MODEL_FOLDER, 'svd_model.pkl')
