"""
Visualisation of scaffolds and augmented molecules
"""
from rdkit.Chem import Draw
from rdkit import Chem
import random

def visualise_smiles(smiles, save_image=False, save_path="temp.png"):
    mol = Chem.MolFromSmiles(smiles)
    Draw.ShowMol(mol)
    if save_image:
        Draw.MolToFile(mol,save_path) 

def add_atoms_to_scaffold(smiles_molecule, mod1="CC", mod2="CCS"):
    scaffold = Chem.MolFromSmiles(smiles_molecule)
    scaffold_atom_num = scaffold.GetNumAtoms()

    mod1, mod2 = Chem.MolFromSmiles(mod1), Chem.MolFromSmiles(mod2)
    combo1 = Chem.CombineMols(scaffold, mod1)
    combo2 = Chem.CombineMols(scaffold, mod2)
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

    return Chem.MolToSmiles(mol1), Chem.MolToSmiles(mol2)

if __name__ == "__main__":
    # smiles_molecule = "c1nccc2n1ccc2"
    smiles_molecule = "C1=CC=CC=C1"
    visualise_smiles(smiles_molecule, save_image=True, save_path='benzene.png')
    mol1, mol2 = add_atoms_to_scaffold(smiles_molecule)
    print(mol1, mol2)
    # visualise_smiles(mol1, save_image=True, save_path='temp_1.png')
    # visualise_smiles(mol2, save_image=True, save_path='temp_2.png')