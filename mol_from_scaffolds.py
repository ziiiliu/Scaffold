import csv
import random
import chemprop
from cv2 import add
from matplotlib.pyplot import sca
from dataset.open_smiles import smiles
from rdkit import Chem

open_smiles_list = list(smiles.values())

def read_smiles(data_path, exclude_header=True, txt_file=False):
    smiles_data = []
    if txt_file:
        with open(data_path) as txt_file:
            for i, smiles in enumerate(txt_file):
                if i % 100000 == 0:
                    print(f"Data Loading Progress: {i}")
                smiles_data.append(smiles)
    else:
        with open(data_path) as csv_file:
            csv_reader = csv.reader(csv_file, delimiter=',')
            for i, row in enumerate(csv_reader):
                if exclude_header and i == 0: continue
                smiles = row[-1]
                smiles_data.append(smiles)
    return smiles_data


def get_scaffold_list(data_files, txt_file=False):

    # Single data file
    if type(data_files) == str:
        smiles_data = read_smiles(data_path=data_files, txt_file=txt_file)
    elif type(data_files) == list:
        smiles_data = []
        for data_file in data_files:
            smiles_data += read_smiles(data_path=data_file, txt_file=txt_file)
    
    size = len(smiles_data)
    scaffolds = set()
    for i, smiles in enumerate(smiles_data):
        if i % 100000 == 0:
            print(f"Scaffold curation progress: {i}/{size}")
        try:
            scaffold = chemprop.data.scaffold.generate_scaffold(smiles)
        except ValueError as e:
            print(e)
            continue
        scaffolds.add(scaffold)
    return list(scaffolds)

def add_atoms_to_scaffold(scaffold_smile, num_pairs=2, output_mol=False):
    """
    function for augmenting the scaffolds by adding atoms/subgraphs to the scaffold without changing the 
    scaffold type. 

    Parameters
    ----------
    num_pairs: int
        Number of random pairs to generate from the scaffold
    output_graph: bool
        True -> returns Mol
        False -> returns SMILE strings
    """
    pairs = []
    ind = 0
    scaffold_mol = Chem.MolFromSmiles(scaffold_smile)
    scaffold_atom_num = scaffold_mol.GetNumAtoms()
    while ind < num_pairs:
        samples = random.sample(open_smiles_list, 2)
        mod1, mod2 = samples
        mod1, mod2 = Chem.MolFromSmiles(mod1), Chem.MolFromSmiles(mod2)
        
        combo1 = Chem.CombineMols(scaffold_mol, mod1)
        combo2 = Chem.CombineMols(scaffold_mol, mod2)
        editable_combo1 = Chem.EditableMol(combo1)
        editable_combo2 = Chem.EditableMol(combo2)

        combo1_atom_num = combo1.GetNumAtoms()
        combo2_atom_num = combo2.GetNumAtoms()

        scaffold_atom_ind = random.randint(0, scaffold_atom_num-1)
        combo1_atom_ind = random.randint(scaffold_atom_num, combo1_atom_num-1)
        combo2_atom_ind = random.randint(scaffold_atom_num, combo2_atom_num-1)

        editable_combo1.AddBond(scaffold_atom_ind, combo1_atom_ind, order=Chem.rdchem.BondType.SINGLE)
        editable_combo2.AddBond(scaffold_atom_ind, combo2_atom_ind, order=Chem.rdchem.BondType.SINGLE)

        mol1, mol2 = editable_combo1.GetMol(), editable_combo2.GetMol()
        if output_mol:
            pairs.append((mol1, mol2))
        else:
            pairs.append((Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2)))
        ind += 1
    return pairs

if __name__ == "__main__":

    # data_file = "../MolCLR/data/bbbp/BBBP.csv"
    # scaffold_list = get_scaffold_list(data_files=data_file)
    # print(scaffold_list)


    # Script for generating scaffolds given a long list of molecules
    # data_file = "../MolCLR/data/pubchem-10m-clean.txt"
    # scaffold_list = get_scaffold_list(data_files=data_file, txt_file=True)

    # with open('scaffolds.txt', 'w') as output_file:
    #     for scaffold in scaffold_list:
    #         output_file.write(scaffold + '\n')

    with open('scaffolds.txt', 'r') as scaffolds:
        for scaffold_smile in scaffolds:
            print(scaffold_smile.rstrip())
            if not scaffold_smile.rstrip(): 
                print("Empty string")
                continue
            pairs = add_atoms_to_scaffold(scaffold_smile)
            print(pairs)
            break