## Search for similar documents
### Doc2vec paper: https://cs.stanford.edu/~quocle/paragraph_vector.pdf
### Instruction: Step by step
- Install neccessary libraries and framework:
pip3 install -r requirements.txt
- Train doc2vec model:
python3 train_doc2vec.py
- Train tf-idf model:
python3 train_tfidf.py
- Train svd model with both doc2vec & TF-IDF:
python3 train_svd.py
- Train KNN model with SVD, doc2vec / TF-IDF:
python3 train_knn.py
- After the training phase, we can use the trained model to search:
python3 search -i path/to/input/file -k n_neighbor -o path/to/output/files