import json
import fnmatch
import os
import argparse


def find_files(src, fname):
    matches = []
    for root, dirnames, filenames in os.walk(src):
        for filename in fnmatch.filter(filenames, fname):
            matches.append(os.path.join(root, filename))

    return matches


def validate_data(data, metrics, dataset):
    for k,v in data[dataset].items():
        for metric in metrics:
            if metric not in v:
                print("{}, {}".format(metric, v['id'])) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, help='dataset to merge')
    parser.add_argument('--search_space', type=str, help='search space to merge')
    args = parser.parse_args()



    dataset=args.dataset
    search_space=args.search_space
    match = search_space+'--'+dataset
    data_file = f'/home/alex/NASLib/naslib/data/zc_{search_space}.json'


    with open(data_file, 'r') as f:
        data = json.load(f)

    # validate_data(data, ('min_depth', 'max_depth'), dataset)

    files = find_files('naslib/data/zc_benchmarks/', '*.json')

    for file in files:
        components = file.split('/')
        metric = components[-2]
        filename = components[-1]
        print(metric, filename)
        if match in filename:
            with open(file) as f:
                zc_benchmarks = json.load(f)
            for entry in zc_benchmarks:
                if entry['arch'] in data[dataset]:
                    data[dataset][entry['arch']][metric] = entry[metric]
                else:
                    raise ValueError("benchmarked architecture not present in dataset")
            # print(entry['idx'])
    # print(data[dataset]['(4, 0, 3, 1, 4, 3)'])
    results_file = f'naslib/data/zc_{search_space}.json'
    with open(results_file, 'w') as f:
        json.dump(data, f)
