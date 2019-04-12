from utils import BaseSvdDoc2vecFunctions
from utils import BaseSvdTfidfFunctions


# Create + train + save svd model with Doc2vec
base_svd_doc2vec = BaseSvdDoc2vecFunctions()
base_svd_doc2vec.fit_svd_model()

# Create + train + save svd model with TF-IDF
base_svd_tfidf = BaseSvdTfidfFunctions()
base_svd_tfidf.fit_svd_model()