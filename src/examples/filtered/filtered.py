from __future__ import print_function
from functools import lru_cache
from scipy.sparse import csr_matrix
import numpy as np
# import falconn
import timeit
import math
import pdb
from collections import defaultdict

cache_dict = {}

use_compute_probes = False

def datapoint_fulfills_constrains(metadata: list[int], filter_metadata: list[int]):
    for i in filter_metadata.indices:
        if metadata[i] != 1:
            return False
    return True

def euclidean_distance(a, b):
    return np.linalg.norm(a - b)

def filter_dataset(dataset, dataset_metadata, filter_metadata):
    new_dataset = np.ndarray(shape=(0, dataset.shape[1]), dtype=dataset.dtype)
    if filter_metadata.toarray().tobytes() in cache_dict:
        return cache_dict[filter_metadata.toarray().tobytes()]
    # breakpoint()
    for i, metadata in enumerate(dataset_metadata.toarray()):
        if datapoint_fulfills_constrains(metadata, filter_metadata):
            new_dataset.append(dataset[i])
    cache_dict[filter_metadata.toarray().tobytes()] = new_dataset
    return new_dataset

def write_sparse_matrix(mat, fname):
    """ write a CSR matrix in the spmat format """
    with open(fname, "wb") as f:
        sizes = np.array([mat.shape[0], mat.shape[1], mat.nnz], dtype='int64')
        sizes.tofile(f)
        indptr = mat.indptr.astype('int64')
        indptr.tofile(f)
        mat.indices.astype('int32').tofile(f)
        mat.data.astype('float32').tofile(f)

def read_sparse_matrix_fields(fname):
    """ read the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
        indptr = np.fromfile(f, dtype='int64', count=nrow + 1)
        assert nnz == indptr[-1]
        indices = np.fromfile(f, dtype='int32', count=nnz)
        assert np.all(indices >= 0) and np.all(indices < ncol)
        data = np.fromfile(f, dtype='float32', count=nnz)
        return data, indices, indptr, ncol

def mmap_sparse_matrix_fields(fname):
    """ mmap the fields of a CSR matrix without instanciating it """
    with open(fname, "rb") as f:
        sizes = np.fromfile(f, dtype='int64', count=3)
        nrow, ncol, nnz = sizes
    ofs = sizes.nbytes
    indptr = np.memmap(fname, dtype='int64', mode='r', offset=ofs, shape=nrow + 1)
    ofs += indptr.nbytes
    indices = np.memmap(fname, dtype='int32', mode='r', offset=ofs, shape=nnz)
    ofs += indices.nbytes
    data = np.memmap(fname, dtype='float32', mode='r', offset=ofs, shape=nnz)
    return data, indices, indptr, ncol

def read_sparse_matrix(fname, do_mmap=False):
    """ read a CSR matrix in spmat format, optionally mmapping it instead """
    if not do_mmap:
        data, indices, indptr, ncol = read_sparse_matrix_fields(fname)
    else:
        data, indices, indptr, ncol = mmap_sparse_matrix_fields(fname)

    return csr_matrix((data, indices, indptr), shape=(len(indptr) - 1, ncol))


if __name__ == '__main__':
    dataset_file = './dataset/data_100000_50'
    dataset_metadata_file = './dataset/data_metadata_100000_50'
    # we build only 50 tables, increasing this quantity will improve the query time
    # at a cost of slower preprocessing and larger memory footprint, feel free to
    # play with this number
    number_of_tables = 10

    print('Reading the dataset')

    dtype = "float32"
    n, d = map(int, np.fromfile(dataset_file, dtype="uint32", count=2))

    dataset = np.memmap(dataset_file, dtype=dtype, mode="r+", offset=8, shape=(n, d))
    dataset_metadata = read_sparse_matrix(dataset_metadata_file, do_mmap=True)
    print('Done')

    # It's important not to use doubles, unless they are strictly necessary.
    # If your dataset consists of doubles, convert it to floats using `astype`.
    # assert dataset.dtype == np.float32

    # Normalize all the lenghts, since we care about the cosine similarity.

    # Choose random data points to be queries.
    # print('Generating queries')
    # np.random.seed(4057218)
    # np.random.shuffle(dataset)
    # queries = dataset[len(dataset) - number_of_queries:]
    # dataset = dataset[:len(dataset) - number_of_queries]
    # print('Done')
    #

    # Fetch queries from file
    queries_file = './dataset/queries_1000_50'
    queries_metadata_file = './dataset/queries_metadata_100000_50'

    print('Reading queries')
    dtype = "float32"
    n, d = map(int, np.fromfile(queries_file, dtype="uint32", count=2))

    queries = np.memmap(queries_file, dtype=dtype, mode="r+", offset=8, shape=(n, d))

    queries_metadata = read_sparse_matrix(queries_metadata_file, do_mmap=True)
    mydic = defaultdict(lambda: set())
    # breakpoint()
    # metadata = dict(dataset_metadata.tolil().items())

    for idx, el in dict(dataset_metadata.todok().items()).keys():
        mydic[idx].add(el)

    print('Done')


    # Perform linear scan using NumPy to get answers to the queries.
    print('Solving queries using linear scan')
    t1 = timeit.default_timer()
    answers = []
    for i in range(len(queries)):
        best_dist = -1
        best_idx = -1
        # breakpoint()
        if i % 10 == 0:
            print('Processing query {}'.format(i))
        query = queries[i]
        start,end = (0, 50000) if i >= 500 else (50000, 100000)
        # print(datapoint_fulfills_constrains(dataset_metadata[start].toarray()[0], queries_metadata[i]))
        for point_idx in range(start,end):
            # if j % 10000 == 0:
            #     print('Processing point {}'.format(j))
            if datapoint_fulfills_constrains(dataset_metadata[point_idx].toarray()[0], queries_metadata[i]):
                if best_dist == -1 or euclidean_distance(query, dataset[point_idx]) < best_dist:
                    best_dist = euclidean_distance(query, dataset[point_idx])
                    best_idx = point_idx
        answers.append(best_idx)
        print('Best idx: {}, Best dist: {}'.format(best_idx, best_dist))
    with open('./answers.py', 'w') as f:f.write(repr(answers))

    # print('Normalizing the dataset')
    # dataset /= np.linalg.norm(dataset, axis=1).reshape(-1, 1)
    # print('Done')

    # t2 = timeit.default_timer()
    # print('Done')
    # print('Linear scan time: {} per query'.format((t2 - t1) / float(
    #     len(queries))))

    # # Center the dataset and the queries: this improves the performance of LSH quite a bit.
    # print('Centering the dataset and queries')
    # center = np.mean(dataset, axis=0)
    # dataset -= center
    # queries -= center
    # print('Done')

    # params_cp = falconn.LSHConstructionParameters()
    # params_cp.dimension = len(dataset[0])
    # params_cp.lsh_family = falconn.LSHFamily.CrossPolytope
    # params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
    # params_cp.l = number_of_tables
    # # we set one rotation, since the data is dense enough,
    # # for sparse data set it to 2
    # params_cp.num_rotations = 1
    # params_cp.seed = 5721840
    # # we want to use all the available threads to set up
    # params_cp.num_setup_threads = 0
    # params_cp.storage_hash_table = falconn.StorageHashTable.BitPackedFlatHashTable
    # # we build 18-bit hashes so that each table has
    # # 2^18 bins; this is a good choise since 2^18 is of the same
    # # order of magnitude as the number of data points
    # falconn.compute_number_of_hash_functions(18, params_cp)

    # print('Constructing the LSH table')
    # t1 = timeit.default_timer()
    # table = falconn.LSHIndex(params_cp)
    # table.setup(dataset, mydic)
    # t2 = timeit.default_timer()
    # print('Done')
    # print('Construction time: {}'.format(t2 - t1))

    # query_object = table.construct_query_object()

    # # find the smallest number of probes to achieve accuracy 0.9
    # # using the binary search
    # print('Choosing number of probes')
    # number_of_probes = number_of_tables

    # def evaluate_number_of_probes(number_of_probes):
    #     query_object.set_num_probes(number_of_probes)
    #     score = 0
    #     for (i, query) in enumerate(queries):
    #         if answers[i] in query_object.get_candidates_with_duplicates(
    #                 query):
    #             score += 1
    #     return float(score) / len(queries)

    # while use_compute_probes:
    #     accuracy = evaluate_number_of_probes(number_of_probes)
    #     print('{} -> {}'.format(number_of_probes, accuracy))
    #     if accuracy >= 0.9:
    #         break
    #     number_of_probes = number_of_probes * 2
    # if number_of_probes > number_of_tables:
    #     left = number_of_probes // 2
    #     right = number_of_probes
    #     while right - left > 1:
    #         number_of_probes = (left + right) // 2
    #         accuracy = evaluate_number_of_probes(number_of_probes)
    #         print('{} -> {}'.format(number_of_probes, accuracy))
    #         if accuracy >= 0.9:
    #             right = number_of_probes
    #         else:
    #             left = number_of_probes
    #     number_of_probes = right

    # query_object.set_num_probes(number_of_probes)
    # print('Done')
    # print('{} probes'.format(number_of_probes))

    # # final evaluation
    # t1 = timeit.default_timer()
    # score = 0
    # res = 0
    # right = 0
    # for (i, query) in enumerate(queries):
    #     res = query_object.find_nearest_neighbor(query, queries_metadata[i].indices)
    #     real_point = None
    #     # for result in res:
    #         # breakpoint()
    #         # if datapoint_fulfills_constrains(dataset_metadata[result].toarray()[0], queries_metadata[i]):
    #             # real_point = result
    #             # break
    #     if (i % 100 == 0):
    #         print('Processing query {}'.format(i))
    #     # breakpoint()
    #     # print('Res object: {}, answer: {}'.format(res, answers[i]))
    #     # print('Fulfills constraints: {}'.format(datapoint_fulfills_constrains(dataset_metadata[res].toarray()[0], queries_metadata[i])))
    #     if datapoint_fulfills_constrains(dataset_metadata[res].toarray()[0], queries_metadata[i]):
    #         right += 1
    #     if res == answers[i]: # find_k_nearest_neighbors allows us to find multiple and not just one nearest neighbor
    #         score += 1
    # t2 = timeit.default_timer()

    # print('Query time: {}'.format((t2 - t1) / len(queries)))
    # print('Precision: {}'.format(float(score) / len(queries)))
    # print('Right: {}'.format(float(right) / len(queries)))
