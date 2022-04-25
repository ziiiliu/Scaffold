import csv
import chemprop
import collections
import numpy as np

def read_smiles(data_path, exclude_header=True):
    smiles_data = []
    with open(data_path) as csv_file:
        csv_reader = csv.reader(csv_file, delimiter=',')
        for i, row in enumerate(csv_reader):
            if exclude_header and i == 0: continue
            if not row: continue
            smiles = row[0]
            smiles_data.append(smiles)
    return smiles_data

def get_scaffold_dict(data_file):
    scaffolds = []
    scaf_dict = collections.defaultdict(list)

    smiles_data = read_smiles(data_path=data_file)
    for i, smiles in enumerate(smiles_data):
        try:
            scaffold = chemprop.data.scaffold.generate_scaffold(smiles)
        except ValueError as e:
            print(e)
            scaffolds.append(np.nan)
            continue
        scaf_dict[scaffold].append(i)
        scaffolds.append(scaffold)
    return scaf_dict

if __name__ == "__main__":

    data_file = "../MolCLR/data/bbbp/BBBP.csv"
    scaf_dict = get_scaffold_dict(data_file=data_file)
    print(scaf_dict.keys())
