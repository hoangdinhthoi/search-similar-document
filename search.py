from utils import BaseKNNFunction
import numpy as np
from settings import file_list
import argparse

base_knn = BaseKNNFunction()
base_knn.load_knn()
parser = argparse.ArgumentParser(
    description='Search for similar document from text database')
parser.add_argument(
    "-i",
    "--input",
    type=str,
    nargs="?",
    help="The input file",
)
parser.add_argument(
    "-k",
    "--n_neighbors",
    type=int,
    nargs="?",
    help="The number of output similar documents.",
    default=10
)

args = parser.parse_args()
with open(args.input, 'r') as f:
    document = f.read()
distance, index = base_knn.search_knn(document, args.n_neighbors)
index = np.squeeze(index).tolist()
distance = np.squeeze(distance).tolist()
for i in range(args.n_neighbors):
    print (file_list[index[i]])

    
