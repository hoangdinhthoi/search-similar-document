from utils import BaseDoc2vecFunction


# Create + train + save doc2vec model
base_doc2vec = BaseDoc2vecFunction()
doc2vec_model = base_doc2vec.train_doc2vec_model()
