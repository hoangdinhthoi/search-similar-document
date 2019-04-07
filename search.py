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
    default="./result.txt"
)
args = parser.parse_args()
if args.input_file is None:
    print("Please specify the path for input document")
    print("Eg. python3 search.py -i path/to/file")
else:
    with open(args.input_file, 'r') as f:
        document = f.read()
    distance, index = base_knn.search_knn(document, args.n_neighbors)
    index = np.squeeze(index).tolist()
    distance = np.squeeze(distance).tolist()

    result_text = ''
    result_text += 'Top {} similar document(s) of input file {}:\n'.format(
        args.n_neighbors, args.input_file)
    result_text += '_' * 50 + '\n'
    result_text += 'STT       ' + 'File name' + ' ' * 17 + 'Distance\n'
    for i in range(args.n_neighbors):
        file_name = '/'.join(file_list[index[i]].split('/')[-3:])
        result_text += str(i+1) + ' ' * (5-len(str(i+1))) + file_name
        result_text += ' ' * 10 + str(distance[i])[:6] + '\n'
    result_text += '_' * 50 + '\n'
    with open(args.output_file, 'w') as f:
        f.write(result_text)
