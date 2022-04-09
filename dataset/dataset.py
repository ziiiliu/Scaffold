"""
Script adapted from TODO
"""

import os
import csv
import math
import time
import random
import networkx as nx
import numpy as np
from copy import deepcopy

import torch
import torch.nn.functional as F
# from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import SubsetRandomSampler
import torchvision.transforms as transforms

from torch_scatter import scatter
from torch_geometric.data import Data, Dataset, DataLoader

import rdkit
from rdkit import Chem
from rdkit.Chem.rdchem import HybridizationType
from rdkit.Chem.rdchem import BondType as BT
from rdkit.Chem import AllChem

from cluster_scaffolds import get_scaffold_dict
from mol_from_scaffolds import read_smiles, add_atoms_to_scaffold

ATOM_LIST = list(range(0,119))
CHIRALITY_LIST = [
    Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
    Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
    Chem.rdchem.ChiralType.CHI_OTHER
]
BOND_LIST = [
    BT.SINGLE, 
    BT.DOUBLE, 
    BT.TRIPLE, 
    BT.AROMATIC
]
BONDDIR_LIST = [
    Chem.rdchem.BondDir.NONE,
    Chem.rdchem.BondDir.ENDUPRIGHT,
    Chem.rdchem.BondDir.ENDDOWNRIGHT
]

class PairedDataset(Dataset):
    def __init__(self, data_path, random_seed=1, augmentation="clustered_scaffold", scaffold_list_path=None):
        super(Dataset, self).__init__()
        
        self.augmentation = augmentation
        if augmentation == "clustered_scaffold":
            self.smiles_data = read_smiles(data_path)
            self.smiles_dict = get_scaffold_dict(data_file=data_path)
            self.x1s, self.x2s = self.generate_paired_input_indices(self.smiles_dict, random_seed)
        elif augmentation == "generated":
            if not scaffold_list_path:
                raise ValueError("Scaffold list txt file not provided.")
            self.mol1s, self.mol2s = self.generate_paired_input_mols_from_txt(txt_file_path=scaffold_list_path)
        else:
            raise ValueError("Augmentation type not recognised")

    def generate_paired_input_mols_from_txt(self, txt_file_path, random_seed=1, max_size=10000):
        """
        For Augmentation approach 2: Given a list of scaffolds in smiles strings, we augment each scaffold by
        adding random atoms/molecules to it without changing the scaffold type
        -----------
        Params
        ----------
        max_size: int
            Maximum number of pairs of inputs to return. For very large scaffold datasets, we might want to 
            limit the size of the dataset. 
        """
        random.seed(random_seed)
        mol1s, mol2s = [], []

        with open(txt_file_path, 'r') as scaffolds:
            for scaffold_smile in scaffolds:
                if not scaffold_smile.rstrip(): 
                    print("Empty string")
                    continue
                pairs = add_atoms_to_scaffold(scaffold_smile, num_pairs=5, output_mol=True)
                mol1s_temp, mol2s_temp = list(zip(*pairs))
                
                mol1s += list(mol1s_temp)
                mol2s += list(mol2s_temp)

                if len(mol1s) > max_size:
                    break
        assert len(mol1s) == len(mol2s)
        return mol1s, mol2s


    def generate_paired_input_indices(self, smiles_dict, random_seed=1):
        random.seed(random_seed)
        x1s, x2s = [], []
        for scaffold, mols in smiles_dict.items():
            # this condition checks for empty scaffold
            if not scaffold:
                continue
            size = len(mols)
            if size <= 1: continue
            random.shuffle(mols)
            x1s += mols[:size//2]
            x2s += mols[size//2:size-size%2]
            assert len(x1s) == len(x2s)
        return x1s, x2s

    def __getitem__(self, index):

        if self.augmentation == "clustered_scaffold":
            mol1_ind, mol2_ind = self.x1s[index], self.x2s[index]
            # print(self.smiles_data[mol1_ind],self.smiles_data[mol2_ind])
            mol1, mol2 = Chem.MolFromSmiles(self.smiles_data[mol1_ind]), Chem.MolFromSmiles(self.smiles_data[mol2_ind])
            # print(type(mol1), type(mol2))
        elif self.augmentation == "generated":
            mol1, mol2 = self.mol1s[index], self.mol2s[index]
        N1, N2 = mol1.GetNumAtoms(), mol2.GetNumAtoms()
        M1, M2 = mol1.GetNumBonds(), mol2.GetNumBonds()

        type_idx_1 = []
        chirality_idx_1 = []
        atomic_number_1 = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        
        # constructing node feature for mol1
        for atom in mol1.GetAtoms():
            type_idx_1.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx_1.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number_1.append(atom.GetAtomicNum())
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(type_idx_1, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx_1, dtype=torch.long).view(-1,1)
        x_1 = torch.cat([x1, x2], dim=-1)
        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)

        row_1, col_1, edge_feat_1 = [], [], []
        for bond in mol1.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row_1 += [start, end]
            col_1 += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat_1.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat_1.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index_1 = torch.tensor([row_1, col_1], dtype=torch.long)
        edge_attr_1 = torch.tensor(np.array(edge_feat_1), dtype=torch.long)


        type_idx_2 = []
        chirality_idx_2 = []
        atomic_number_2 = []
        # aromatic = []
        # sp, sp2, sp3, sp3d = [], [], [], []
        # num_hs = []
        
        # constructing node feature for mol2
        for atom in mol2.GetAtoms():
            type_idx_2.append(ATOM_LIST.index(atom.GetAtomicNum()))
            chirality_idx_2.append(CHIRALITY_LIST.index(atom.GetChiralTag()))
            atomic_number_2.append(atom.GetAtomicNum())
            # aromatic.append(1 if atom.GetIsAromatic() else 0)
            # hybridization = atom.GetHybridization()
            # sp.append(1 if hybridization == HybridizationType.SP else 0)
            # sp2.append(1 if hybridization == HybridizationType.SP2 else 0)
            # sp3.append(1 if hybridization == HybridizationType.SP3 else 0)
            # sp3d.append(1 if hybridization == HybridizationType.SP3D else 0)

        # z = torch.tensor(atomic_number, dtype=torch.long)
        x1 = torch.tensor(type_idx_2, dtype=torch.long).view(-1,1)
        x2 = torch.tensor(chirality_idx_2, dtype=torch.long).view(-1,1)
        x_2 = torch.cat([x1, x2], dim=-1)
        # x2 = torch.tensor([atomic_number, aromatic, sp, sp2, sp3, sp3d, num_hs],
        #                     dtype=torch.float).t().contiguous()
        # x = torch.cat([x1.to(torch.float), x2], dim=-1)

        row_2, col_2, edge_feat_2 = [], [], []
        for bond in mol2.GetBonds():
            start, end = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            row_2 += [start, end]
            col_2 += [end, start]
            # edge_type += 2 * [MOL_BONDS[bond.GetBondType()]]
            edge_feat_2.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])
            edge_feat_2.append([
                BOND_LIST.index(bond.GetBondType()),
                BONDDIR_LIST.index(bond.GetBondDir())
            ])

        edge_index_2 = torch.tensor([row_2, col_2], dtype=torch.long)
        edge_attr_2 = torch.tensor(np.array(edge_feat_2), dtype=torch.long)

        data_1 = Data(x=x_1, edge_index=edge_index_1, edge_attr=edge_attr_1)
        data_2 = Data(x=x_2, edge_index=edge_index_2, edge_attr=edge_attr_2)

        return data_1, data_2

    def __len__(self):
        if self.augmentation == "clustered_scaffold":
            return len(self.x1s)
        elif self.augmentation == "generated":
            return len(self.mol1s)

class PairedDatasetWrapper(object):
    def __init__(self, batch_size, num_workers, valid_size, data_path, augmentation="clustered_scaffold"):
        super(object, self).__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.valid_size = valid_size
        self.augmentation = augmentation

    def get_data_loaders(self):
        if self.augmentation == "clustered_scaffold":
            train_dataset = PairedDataset(data_path=self.data_path)
        elif self.augmentation == "generated":
            train_dataset = PairedDataset(data_path=self.data_path, augmentation=self.augmentation, scaffold_list_path=self.data_path)
        train_loader, valid_loader = self.get_train_validation_data_loaders(train_dataset)
        return train_loader, valid_loader

    def get_train_validation_data_loaders(self, train_dataset):
        # obtain training indices that will be used for validation
        num_train = len(train_dataset)
        indices = list(range(num_train))
        np.random.shuffle(indices)

        split = int(np.floor(self.valid_size * num_train))
        print('split: ', split)
        print('num_train: ', num_train)
        train_idx, valid_idx = indices[split:], indices[:split]

        # define samplers for obtaining training and validation batches
        train_sampler = SubsetRandomSampler(train_idx)
        valid_sampler = SubsetRandomSampler(valid_idx)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=train_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        valid_loader = DataLoader(train_dataset, batch_size=self.batch_size, sampler=valid_sampler,
                                  num_workers=self.num_workers, drop_last=True)

        return train_loader, valid_loader