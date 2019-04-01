import os
import pickle
import numpy as np
import unicodedata as ud
from tqdm import tqdm
from pyvi import ViTokenizer
from sklearn.decomposition import TruncatedSVD
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
        '''Tokeinize a document into list of tokens
        # Arguments
            document: input document for tokenizing
        # Return
            list of tokens of the input document
        '''
        tokens = ViTokenizer.tokenize(document)
        return tokens.split(' ')


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
        self.tagged_document = self.create_tagged_document()

    def create_tagged_document(self):
        '''Create tagged document before training doc2vec algorithm
        # Arguments
            document_folder: folder of the training data
        # Return
            list of tagged document
        '''
        tagged_document = []
        file_list = os.listdir(FULL_DATA_FOLDER)
        for index, f in tqdm(enumerate(file_list)):
            with open(os.path.join(FULL_DATA_FOLDER, f), 'r') as fr:
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
        self.doc2vec_model.build_vocab(self.tagged_document)
        print('Begin training')
        print('_'*100)
        for epoch in range(MAX_EPOCHS):
            print('Iteration', epoch + 1)
            self.doc2vec_model.train(
                self.tagged_document,
                total_examples=self.doc2vec_model.corpus_count,
                epochs=self.doc2vec_model.iter
            )
            self.doc2vec_model.alpha -= 0.0001
            self.doc2vec_model.min_alpha = self.doc2vec_model.alpha
        self.save_doc2vec_model()
        print('_'*100)
        print('Finish training after', MAX_EPOCHS, 'epochs')
        print('Save doc2vec model at', DOC2VEC_MODEL_PATH)
        return self.doc2vec_model

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

    @staticmethod
    def load_doc2vec_model():
        '''Load doc2vec mode from file
        '''
        return Doc2Vec.load(DOC2VEC_MODEL_PATH)

    @staticmethod
    def create_document_embedding(doc2vec_model, document):
        '''Generate document embedding from doc2vec model
        # Arguents
            doc2vec_model: a trained doc2vec model
            document: input document to generate embedding
        # Return
            embedding of the input document
        '''
        document = BaseTextFunction.process_document(document)
        document = BaseTextFunction.tokenize(document)
        return doc2vec_model.infer_vector(document)


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

    def create_svd_model(self):
        '''Create SVD from parameters in setting file
        '''
        return TruncatedSVD(
            n_components=SVD_OUTPUT_FEATURE,
            algorithm=SVD_SOLVER,
            n_iter=SVD_ITERATION
        )

    def fit_svd_model(self, doc2vec_model):
        '''Fit data to svd model
        '''
        list_arr = []
        file_list = os.listdir(FULL_DATA_FOLDER)
        for f in tqdm(file_list):
            with open(os.path.join(FULL_DATA_FOLDER, f), 'r') as fr:
                content = BaseTextFunction.process_document(fr.read())
                words = BaseTextFunction.tokenize(content)
                embedding_vector = doc2vec_model.infer_vector(words)
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

    @staticmethod
    def load_svd_model():
        '''Load svd model from file
        '''
        with open(SVD_MODEL_PATH, 'rb') as f:
            return pickle.load(f)
