from utils import BaseKNNDoc2vecFunction
from utils import BaseKNNTfidfFunction


base_knn_doc2vec = BaseKNNDoc2vecFunction()
base_knn_doc2vec.fit_knn()

base_knn_tfidf = BaseKNNTfidfFunction()
base_knn_tfidf.fit_knn()
