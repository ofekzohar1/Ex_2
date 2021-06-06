import sys
import numpy as np
import pandas as pd
import mykmeanssp

DEFAULT_ITER = 300
MIN_ARGUMENTS = 3
FIRST_FILE_INDEX_IN_ARGV = 2
SECOND_FILE_INDEX_IN_ARGV = 3
COMMA = ','


def main():
    k, max_iter = validate_and_assign_input_user()
    df_of_vectors = build_vectors_panda()
    number_of_vectors, dimensions = df_of_vectors.shape
    if k >= number_of_vectors:
        print(f"K must be smaller than the number of vectors: K = {k}, number of vectors = {number_of_vectors}")
        exit()  # End program k >= n

    list_random_init_centrals_indexes = choose_random_centrals(df_of_vectors, k, number_of_vectors)
    print(*list_random_init_centrals_indexes, sep=COMMA)
    list_of_vectors = df_of_vectors.values.tolist()
    final_centroids_list = mykmeanssp.fit(k, max_iter, dimensions, number_of_vectors, list_random_init_centrals_indexes,
                                          list_of_vectors)
    print_centrals(final_centroids_list)


def validate_and_assign_input_user():
    if len(sys.argv) < MIN_ARGUMENTS + 1:
        print(f"Amount of arguments should be at least {MIN_ARGUMENTS}: amount of arguments = {len(sys.argv) - 1}")
        exit()  # End program, min arguments
    if (not sys.argv[1].isdigit()) or int(sys.argv[1]) < 1:
        print(f"K input has to be a number and should exceed 0: k = {sys.argv[1]}")
        exit()  # End program, not valid k
    k = int(sys.argv[1])
    max_iter = DEFAULT_ITER
    if len(sys.argv) > MIN_ARGUMENTS + 1:
        if (sys.argv[2].isdigit()) and int(sys.argv[2]) > 0:
            max_iter = int(sys.argv[2])
        else:
            print(f"max_iter input has to be a number and should exceed 0: max_iter = {sys.argv[2]}")
            exit()  # End program, not valid max_iter
    return k, max_iter


def build_vectors_panda():
    add_argv_index = 1 if len(sys.argv) > MIN_ARGUMENTS + 1 else 0
    try:
        pd_1 = pd.read_csv(sys.argv[FIRST_FILE_INDEX_IN_ARGV + add_argv_index], header=None)
        pd_2 = pd.read_csv(sys.argv[SECOND_FILE_INDEX_IN_ARGV + add_argv_index], header=None)
    except FileNotFoundError:
        print("File not accessible")
        exit()
    pd_1.rename(columns={list(pd_1)[0]: 'id'}, inplace=True)  # renaming both first columns for the merge
    pd_2.rename(columns={list(pd_2)[0]: 'id'}, inplace=True)
    df = pd.merge(pd_1, pd_2, how='inner', on='id')
    df.sort_values('id', inplace=True)
    df = df.iloc[:, 1:]
    return df


# receives the vectors pandas and amount of k clusters and builds a list of clusters
# returns a list of clusters
def choose_random_centrals(df_of_vectors, k, num_of_vectors):
    np.random.seed(0)
    list_random_init_centrals_indexes = [np.random.choice(num_of_vectors)]
    np_of_vectors = df_of_vectors.to_numpy()
    for i in range(k - 1):
        np_subtract = np_of_vectors - np_of_vectors[list_random_init_centrals_indexes[i]]
        np_norms = np.linalg.norm(np_subtract, axis=1) ** 2
        np_min_norms = np.minimum(np_min_norms, np_norms) if i > 0 else np_norms
        np_prob = np_min_norms / np_min_norms.sum()
        list_random_init_centrals_indexes.append(np.random.choice(num_of_vectors, p=np_prob))
    return list_random_init_centrals_indexes


# prints new central after adjusting for the relevant structure
def print_centrals(final_centroids_list):
    np_centroids = np.round(final_centroids_list, decimals=4)
    for central in np_centroids:
        print(*central, sep=COMMA)


if __name__ == '__main__':
    main()
