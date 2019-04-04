from utils import BaseSVDFunctions
from utils import BaseDoc2vecFunction


# load doc2vec model
doc2vec_model = BaseDoc2vecFunction.load_doc2vec_model()
# Create + train + save svd model
base_svd = BaseSVDFunctions()
base_svd.fit_svd_model(doc2vec_model)
