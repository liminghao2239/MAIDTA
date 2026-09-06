import torch
import numpy as np
from tqdm import tqdm
from rdkit import Chem


CHARPROTSET = {
    "A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
    "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
    "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
    "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25
}


CHARSMISET = {
    "#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
    "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
    "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
    "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
    "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
    "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 55, "a": 56, "c": 57,
    "b": 20, "e": 58, "d": 21, "g": 59, "f": 22, "i": 60, "h": 23, "m": 61,
    "l": 24, "o": 62, "n": 25, "s": 63, "r": 26, "u": 64, "t": 27, "y": 65
}


def one_of_k_encoding(x, allowable_set):
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(x, allowable_set):
    return [x == s for s in allowable_set] if x in allowable_set else [False] * len(allowable_set)


def atom_features(atom):
    return np.array(
        one_of_k_encoding_unk(atom.GetSymbol(), [
            'C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca',
            'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag',
            'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd',
            'In', 'Mn', 'Zr', 'Cr', 'Pt', 'Hg', 'Pb', 'Unknown'
        ]) +
        one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        one_of_k_encoding(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
        [atom.GetIsAromatic()]
    )


def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([
        bt == Chem.rdchem.BondType.SINGLE,
        bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE,
        bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ])


def smiles_to_graph(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    atom_feats = np.array([atom_features(atom) for atom in mol.GetAtoms()])

    edge_index = []
    edge_feats = []
    for bond in mol.GetBonds():
        i = bond.GetBeginAtomIdx()
        j = bond.GetEndAtomIdx()
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_feats.append(bond_features(bond))
        edge_feats.append(bond_features(bond))

    if len(edge_index) == 0:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_feats = torch.zeros((0, 6), dtype=torch.float)
    else:
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        edge_feats = torch.tensor(edge_feats, dtype=torch.float)

    return {
        'x': torch.tensor(atom_feats, dtype=torch.float),
        'edge_index': edge_index,
        'edge_attr': edge_feats
    }


def smiles_to_tokens(smiles, max_len=100):
    tokens = []
    for ch in smiles[:max_len]:
        tokens.append(CHARSMISET.get(ch, 0))
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.tensor(tokens[:max_len], dtype=torch.long)


def protein_to_tokens(seq, max_len=1024):
    tokens = []
    for ch in seq[:max_len]:
        tokens.append(CHARPROTSET.get(ch, 0))
    while len(tokens) < max_len:
        tokens.append(0)
    return torch.tensor(tokens[:max_len], dtype=torch.long)


def prepare_maidta_data(smiles_list, seq_list, labels, esm_dir=None, device='cpu', max_prot_len=1024):
    data_list = []

    if esm_dir is not None:
        import os
        import glob
        esm_features = {}
        for pt_file in glob.glob(os.path.join(esm_dir, '*.pt')):
            prot_id = os.path.basename(pt_file).replace('.pt', '')
            try:
                feat = torch.load(pt_file)
                if 'representations' in feat:
                    esm_features[prot_id] = feat['representations'].mean(dim=1)
                elif 'sequence_repr' in feat:
                    esm_features[prot_id] = feat['sequence_repr']
            except:
                pass
        print(f"Loaded ESM features for {len(esm_features)} proteins")
    else:
        esm_features = None

    for i in tqdm(range(len(smiles_list))):
        smiles = smiles_list[i]
        seq = seq_list[i]
        label = labels[i]

        mol_graph = smiles_to_graph(smiles)
        if mol_graph is None:
            continue

        smiles_tokens = smiles_to_tokens(smiles, max_len=100)
        seq_tokens = protein_to_tokens(seq, max_len=max_prot_len)

        if esm_features is not None:
            prot_id = seq[:10]
            if prot_id in esm_features:
                esm_feat = esm_features[prot_id]
                if isinstance(esm_feat, torch.Tensor):
                    esm_feat = esm_feat.numpy()
                if len(esm_feat) > max_prot_len:
                    esm_feat = esm_feat[:max_prot_len]
                elif len(esm_feat) < max_prot_len:
                    esm_feat = np.pad(esm_feat, ((0, max_prot_len - len(esm_feat)), (0, 0)), 'constant')
                seq_tokens = torch.tensor(esm_feat, dtype=torch.float)
            else:
                seq_tokens = torch.zeros((max_prot_len, 1280), dtype=torch.float)

        drug_data = {
            'smiles_tokens': smiles_tokens,
            'graph_x': mol_graph['x'],
            'graph_edge_index': mol_graph['edge_index'],
            'graph_edge_attr': mol_graph['edge_attr'],
            'y': torch.tensor([label], dtype=torch.float),
            'batch': torch.zeros(mol_graph['x'].size(0), dtype=torch.long)
        }

        prot_data = {
            'seq_tokens': seq_tokens,
            'graph_x': torch.zeros(1, 256),
            'graph_edge_index': torch.zeros((2, 0), dtype=torch.long),
            'graph_edge_attr': torch.zeros((0, 1), dtype=torch.float),
            'batch': torch.zeros(1, dtype=torch.long)
        }

        data_list.append((drug_data, prot_data))

    return data_list