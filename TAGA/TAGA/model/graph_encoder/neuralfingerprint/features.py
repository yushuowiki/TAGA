import autograd.numpy as np
from rdkit import Chem

def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return map(lambda s: x == s, allowable_set)

def one_of_k_encoding_unk(x, allowable_set):
    """Maps inputs not in the allowable set to the last element."""
    if x not in allowable_set:
        x = allowable_set[-1]
    return map(lambda s: x == s, allowable_set)

def atom_features(atom):

    return np.array(list(one_of_k_encoding_unk(atom.GetSymbol(),
                                      ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
                                       'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
                                       'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',    # H?
                                       'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
                                       'Cr', 'Pt', 'Hg', 'Pb','Unknown'])) + 
                    list(one_of_k_encoding_unk(atom.GetDegree(), [0, 1, 2, 3, 4, 5])) +
                    list(one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4])) +
                    list(one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5])) +
                    [atom.GetIsAromatic()])


# def one_of_k_encoding(x, allowable_set):
#     if x not in allowable_set:
#         raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
#     return map(lambda s: x == s, allowable_set)

# def one_of_k_encoding_unk(x, allowable_set):
#     """Maps inputs not in the allowable set to the last element."""
#     if x not in allowable_set:
#         x = allowable_set[-1]
#     return map(lambda s: x == s, allowable_set)

# def safe_index(l, e):
#     """
#     Return index of element e in list l. If e is not present, return the last index
#     """
#     try:
#         return l.index(e)
#     except:
#         return len(l) - 1

# def atom_features(atom):
#     # get atomic properties
#     symbol = atom.GetSymbol()
#     degree = atom.GetDegree()
#     num_h = atom.GetTotalNumHs()
#     implicit_valence = atom.GetImplicitValence()
    
#     # fault tolerance
#     symbol_features = list(one_of_k_encoding_unk(symbol,
#                                       ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na',
#                                        'Ca', 'Fe', 'As', 'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb',
#                                        'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se', 'Ti', 'Zn', 'H',
#                                        'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr',
#                                        'Cr', 'Pt', 'Hg', 'Pb', 'Rh','Unknown']))  # Rh: 7，Fe: 1，U: 1
    
#     degree_features = list(one_of_k_encoding_unk(degree, 
#                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc']))
    
#     num_h_features = list(one_of_k_encoding_unk(num_h, 
#                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 'misc']))
    
#     implicit_valence_features = list(one_of_k_encoding_unk(implicit_valence, 
#                                       [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc']))
    
#     aromatic_feature = [atom.GetIsAromatic()]
    
#     return np.array(symbol_features + degree_features + num_h_features + 
#                     implicit_valence_features + aromatic_feature)

def bond_features(bond):
    bt = bond.GetBondType()
    return np.array([bt == Chem.rdchem.BondType.SINGLE,
                     bt == Chem.rdchem.BondType.DOUBLE,
                     bt == Chem.rdchem.BondType.TRIPLE,
                     bt == Chem.rdchem.BondType.AROMATIC,
                     bond.GetIsConjugated(),
                     bond.IsInRing()])

def num_atom_features():
    # Return length of feature vector using a very simple molecule.
    m = Chem.MolFromSmiles('CC')
    alist = m.GetAtoms()
    a = alist[0]
    return len(atom_features(a))

def num_bond_features():
    # Return length of feature vector using a very simple molecule.
    simple_mol = Chem.MolFromSmiles('CC')
    Chem.SanitizeMol(simple_mol)
    return len(bond_features(simple_mol.GetBonds()[0]))

