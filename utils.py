import os
import pickle
import numpy as np
import unicodedata as ud
from tqdm import tqdm
from pyvi import ViTokenizer
from sklearn.decomposition import TruncatedSVD
from sklearn.neighbors import NearestNeighbors
from gensim.parsing.preprocessing import strip_numeric
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim.parsing.preprocessing import strip_punctuation
from gensim.parsing.preprocessing import strip_multiple_whitespaces
from settings import ALLOWED_KEYWORDS
from settings import FULL_DATA_FOLDER
from settings import VECTOR_SIZE
from settings import ALPHA
from settings import MIN_ALPHA
from settings import DM
from settings import MIN_COUNT
from settings import DOC2VEC_MODEL_PATH
from settings import TRAINED_MODEL_FOLDER
from settings import MAX_EPOCHS
from settings import SVD_OUTPUT_FEATURE
from settings import SVD_MODEL_PATH
from settings import SVD_SOLVER
from settings import SVD_ITERATION
from settings import FILE_LIST_PATH
from settings import file_list
from settings import stop_words
from settings import KNN_N_NEIGHBORS
from settings import KNN_MODEL_PATH


class BaseTextFunction():
    '''Class for basic processing text function
    # Methods
        to_lower(document)
        remove_space(document)
        remove_special(document)
        remove_number(document)
        processe_document(document)
    '''
    @staticmethod
    def to_lower(document):
        '''Convert document to lower-case document
        # Arguments
            document: input document to be lower-cased
        # Returns
            lower-case document
        '''
        return document.lower()

    @staticmethod
    def remove_space(document):
        '''Remove multiple white space of a document
        # Arguments
            document: input document before removing white space
        # Return
            document after being removed white space
        '''
        return strip_multiple_whitespaces(document)

    @staticmethod
    def remove_special(document):
        '''Remove sepcial characters from a document.
        Eg. comma, semi-colon, question mark, etc
        # Arguments
            document: input document before removing special characters
        # Return
            document after being removed special characters
        '''
        return strip_punctuation(document)

    @staticmethod
    def remove_number(document):
        '''Remove number from a document.
        # Arguments
            document: input document before removing number
        # Return
            document after being removed number
        '''
        return strip_numeric(document)

    @staticmethod
    def process_document(document, *args, **kwargs):
        '''Process a document with desired functionality
        # Return
            document after being processed
        '''
        # These properties must be set by user via keyword arguments
        for kwarg in kwargs:
            if kwarg not in ALLOWED_KEYWORDS:
                raise TypeError('Keyword argument not understood:', kwarg)
        processed_document = ud.normalize('NFC', document)
        if kwargs.get('to_lower', True) is True:
            processed_document = BaseTextFunction.to_lower(document)
        if kwargs.get('remove_special', True) is True:
            processed_document = BaseTextFunction.remove_special(processed_document)
        if kwargs.get('remove_number', True) is True:
            processed_document = BaseTextFunction.remove_number(processed_document)
        if kwargs.get('remove_space', True) is True:
            processed_document = BaseTextFunction.remove_space(processed_document)
        return processed_document

    @staticmethod
    def tokenize(document):
        '''Tokeinize a document into list of tokens and remove stop words
        # Arguments
            document: input document for tokenizing
        # Return
            list of tokens of the input document
        '''
        tokens = ViTokenizer.tokenize(document)
        tokens = tokens.split(' ')
        return [t for t in tokens if t not in stop_words]


class BaseDoc2vecFunction():
    '''Class for basic Doc2Vec functions
    # Methods
        __init__(self)
        create_tagged_document(self)
        create_doc2vec_model(self)
        train_doc2vec_model(self)
        save_doc2vec_model(self)
        load_doc2vec_model(self)
    '''
    def __init__(self):
        self.doc2vec_model = self.create_doc2vec_model()

    def create_tagged_document(self):
        '''Create tagged document before training doc2vec algorithm
        # Arguments
            document_folder: folder of the training data
        # Return
            list of tagged document
        '''
        print('_'*100)
        print('Creating tagged document for training doc2vec')
        tagged_document = []
        for index, f in tqdm(enumerate(file_list)):
            with open(f, 'r') as fr:
                content = BaseTextFunction.process_document(fr.read())
                words = BaseTextFunction.tokenize(content)
                tagged_document.append(TaggedDocument(
                    words=words, tags=str(index)))
        return tagged_document

    def create_doc2vec_model(self):
        '''Create a doc2vec model from gensim with defined parameters in settings file
        # Arguments
        # Return
            doc2vec model
        '''
        return Doc2Vec(
            vector_size=VECTOR_SIZE,
            alpha=ALPHA,
            min_alpha=MIN_ALPHA,
            min_count=MIN_COUNT,
            dm=DM
        )

    def train_doc2vec_model(self):
        '''Train & save doc2vec model
        # Arguments
        # Return
        '''
        tagged_document = self.create_tagged_document()
        self.doc2vec_model.build_vocab(tagged_document)
        print('_'*100)
        print('Begin training')
        for epoch in range(MAX_EPOCHS):
            print('Epoch', epoch + 1)
            self.doc2vec_model.train(
                tagged_document,
                total_examples=self.doc2vec_model.corpus_count,
                epochs=self.doc2vec_model.iter
            )
            self.doc2vec_model.alpha -= 0.0001
            self.doc2vec_model.min_alpha = self.doc2vec_model.alpha
        self.save_doc2vec_model()
        print('_'*100)
        print('Finish training after', MAX_EPOCHS, 'epochs')
        print('Save doc2vec model at', DOC2VEC_MODEL_PATH)

    def save_doc2vec_model(self):
        '''Save a doc2vec model after training
        # Arguments
            model: doc2vec model after training, default
                path to is ./trained_model/doc2vec.model
        # Return
        '''
        if not os.path.exists(TRAINED_MODEL_FOLDER):
            os.mkdir(TRAINED_MODEL_FOLDER)
        self.doc2vec_model.save(DOC2VEC_MODEL_PATH)

    def load_doc2vec_model(self):
        '''Load doc2vec mode from file
        '''
        self.doc2vec_model = Doc2Vec.load(DOC2VEC_MODEL_PATH)

    def create_document_embedding(self, document):
        '''Generate document embedding from doc2vec model
        # Arguents
            doc2vec_model: a trained doc2vec model
            document: input document to generate embedding
        # Return
            embedding of the input document
        '''
        document = BaseTextFunction.process_document(document)
        document = BaseTextFunction.tokenize(document)
        return self.doc2vec_model.infer_vector(
            doc_words=document,
            alpha=ALPHA,
            min_alpha=MIN_ALPHA,
            steps=MAX_EPOCHS*2,
        )


class BaseSVDFunctions():
    '''Class for basic functions of SVD
    # Methods
        __init__(self)
        create_svd_model(self)
        fit_svd_model(self, doc2vec_model)
        save_svd_model(self)
        load_svd_model(self)
    '''
    def __init__(self):
        self.svd_model = self.create_svd_model()
        self.doc2vec_obj = BaseDoc2vecFunction()
        self.doc2vec_obj.load_doc2vec_model()

    def create_svd_model(self):
        '''Create SVD from parameters in setting file
        '''
        return TruncatedSVD(
            n_components=SVD_OUTPUT_FEATURE,
            algorithm=SVD_SOLVER,
            n_iter=SVD_ITERATION
        )

    def fit_svd_model(self):
        '''Fit data to svd model
        '''
        print('Building SVD model')
        list_arr = []
        for f in tqdm(file_list):
            with open(f, 'r') as fr:
                embedding_vector = self.doc2vec_obj.create_document_embedding(fr.read())
                list_arr.append(embedding_vector)
        list_arr = np.array(list_arr)
        self.svd_model.fit(list_arr)
        self.save_svd_model()
        return self.svd_model

    def save_svd_model(self):
        '''Save SVD model into file
        '''
        if not os.path.exists(TRAINED_MODEL_FOLDER):
            os.mkdir(TRAINED_MODEL_FOLDER)
        with open(SVD_MODEL_PATH, 'wb') as f:
            pickle.dump(self.svd_model, f)

    def create_embedding_svd(self, document):
        '''Create svd embedding for a document
        '''
        doc2vec_embedding = self.doc2vec_obj.create_document_embedding(document)
        svd_embedding = self.svd_model.transform(np.array([doc2vec_embedding]))
        return np.squeeze(svd_embedding)

    def load_svd_model(self):
        '''Load svd model from file
        '''
        with open(SVD_MODEL_PATH, 'rb') as f:
            self.svd_model = pickle.load(f)


class BaseKNNFunction():
    '''
    '''
    def __init__(self):
        self.knn_model = self.create_knn()
        self.svd_obj = BaseSVDFunctions()
        self.svd_obj.load_svd_model()

    def create_knn(self):
        return NearestNeighbors(KNN_N_NEIGHBORS)

    def fit_knn(self):
        '''Fit KNN model to data
        '''
        print('Building KNN model')
        list_arr = []
        for f in tqdm(file_list):
            with open(f, 'r') as fr:
                svd_embedding = self.svd_obj.create_embedding_svd(fr.read())
                list_arr.append(svd_embedding)
        list_arr = np.array(list_arr)
        self.knn_model.fit(list_arr)
        self.save_knn_model()
        return self.knn_model

    def save_knn_model(self):
        '''Save KNN model to file
        '''
        with open(KNN_MODEL_PATH, 'wb') as f:
            pickle.dump(self.knn_model, f)

    def load_knn(self):
        '''Load KNN model from file
        '''
        with open(KNN_MODEL_PATH, 'rb') as f:
            self.knn_model = pickle.load(f)

    def search_knn(self, document):
        '''Search for similar documents
        '''
        svd_embedding = self.svd_obj.create_embedding_svd(document)
        result = self.knn_model.kneighbors(svd_embedding)
        return result


def create_file_list():
    '''Create file list
    '''
    file_list = sorted(os.listdir(FULL_DATA_FOLDER))
    with open(FILE_LIST_PATH, 'wb') as f:
        pickle.dump(file_list, f)

if __name__ == '__main__':
    create_file_list()
