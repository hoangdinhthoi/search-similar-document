from utils import BaseKNNDoc2vecFunction
from utils import BaseKNNTfidfFunction
import numpy as np
from settings import file_list
import argparse

base_knn_doc2vec = BaseKNNDoc2vecFunction()
base_knn_doc2vec.load_knn()
base_knn_tfidf = BaseKNNTfidfFunction()
base_knn_tfidf.load_knn()
parser = argparse.ArgumentParser(
    description='Search for similar document from text database')
parser.add_argument(
    "-i",
    "--input_file",
    type=str,
    nargs="?",
    help="The input file for searching.",
)
parser.add_argument(
    "-k",
    "--n_neighbors",
    type=int,
    nargs="?",
    help="The number of similar documents to output.",
    default=10
)
parser.add_argument(
    "-o",
    "--output_file",
    type=str,
    nargs="?",
    help="The output file to write result.",
    default="result.txt"
)
args = parser.parse_args()
if args.input_file is None:
    print("Please specify the path for input document")
    print("Eg. python3 search.py -i path/to/file")
else:
    with open(args.input_file, 'r') as f:
        document = f.read()
    distance_doc2vec, index_doc2vec = base_knn_doc2vec.search_knn(
        document, args.n_neighbors)
    index_doc2vec = np.squeeze(index_doc2vec).tolist()
    distance_doc2vec = np.squeeze(distance_doc2vec).tolist()

    distance_tfidf, index_tfidf = base_knn_tfidf.search_knn(
        document, args.n_neighbors)
    index_tfidf = np.squeeze(index_tfidf).tolist()
    distance_tfidf = np.squeeze(distance_tfidf).tolist()

    result_text = ''
    result_text += 'Top {} similar document(s) of {}:\n'.format(
        args.n_neighbors, args.input_file)
    result_text += '_' * 50 + '\n'
    result_text += 'Doc2vec result\n'
    result_text += '_' * 50 + '\n'
    result_text += 'STT       ' + 'File name' + ' ' * 17 + 'Distance\n'
    for i in range(args.n_neighbors):
        file_name = '/'.join(file_list[index_doc2vec[i]].split('/')[-3:])
        result_text += str(i+1) + ' ' * (5-len(str(i+1))) + file_name
        result_text += ' ' * 10 + str(distance_doc2vec[i])[:6] + '\n'
    result_text += '\n'
    result_text += '_' * 50 + '\n'
    result_text += 'TF-IDF result\n'
    result_text += '_' * 50 + '\n'
    result_text += 'STT       ' + 'File name' + ' ' * 17 + 'Distance\n'
    for i in range(args.n_neighbors):
        file_name = '/'.join(file_list[index_tfidf[i]].split('/')[-3:])
        result_text += str(i+1) + ' ' * (5-len(str(i+1))) + file_name
        result_text += ' ' * 10 + str(distance_tfidf[i])[:6] + '\n'
    with open(args.output_file, 'w') as f:
        f.write(result_text)
