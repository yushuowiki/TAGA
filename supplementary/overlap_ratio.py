from rdkit import Chem
from rdkit.Chem import BRICS

def get_brics_fragments_atom_sets(mol):
    """
    return:atom index set
    """
    frags = BRICS.BRICSDecompose(mol, returnMols=True)
    atom_sets = []

    for frag in frags:
        matches = mol.GetSubstructMatches(frag)
        for match in matches:
            atom_sets.append(set(match))

    return atom_sets

def compute_overlap_ratio(pred_atoms, frag_atoms):
    """
    pred_atoms: set(int)
    frag_atoms: set(int)
    """
    intersection = pred_atoms & frag_atoms
    return len(intersection) / len(pred_atoms)

def best_overlap_for_motif(pred_atoms, brics_atom_sets):
    overlaps = [
        compute_overlap_ratio(pred_atoms, frag_atoms)
        for frag_atoms in brics_atom_sets
    ]
    return max(overlaps) if overlaps else 0.0

def evaluate_motif_quality(smiles, predicted_motifs):
    """
    smiles: str
    predicted_motifs: List[List[int]]
    """
    mol = Chem.MolFromSmiles(smiles)
    brics_atom_sets = get_brics_fragments_atom_sets(mol)

    results = []
    for motif in predicted_motifs:
        pred_atoms = set(motif)
        best_overlap = best_overlap_for_motif(pred_atoms, brics_atom_sets)
        results.append(best_overlap)

    return results
                           
def compute_hit_at_tau(overlaps, tau=0.7):
    hits = [1 if o >= tau else 0 for o in overlaps]
    return sum(hits) / len(hits)