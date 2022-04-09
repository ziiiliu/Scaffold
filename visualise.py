"""
Visualisation of scaffolds and augmented molecules
"""
from rdkit.Chem import Draw
from rdkit import Chem

def visualise_smiles(smiles, save_image=False, save_path="temp.png"):
    mol = Chem.MolFromSmiles(smiles)
    Draw.ShowMol(mol)
    if save_image:
        Draw.MolToFile(mol,save_path) 

if __name__ == "__main__":
    smiles_molecule = "c1nccc2n1ccc2"
    visualise_smiles(smiles_molecule, save_image=False, save_path='temp.png')
    