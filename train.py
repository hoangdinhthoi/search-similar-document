from utils import BaseDoc2vecFunction
from utils import BaseSVDFunctions

'''
Phase 1: Create + train + save doc2vec model
Phase 2: Create + fit + save svd model
'''

base_doc2vec = BaseDoc2vecFunction()
base_svd = BaseSVDFunctions()

doc2vec_model = base_doc2vec.train_doc2vec_model()
base_svd.fit_svd_model(doc2vec_model)
