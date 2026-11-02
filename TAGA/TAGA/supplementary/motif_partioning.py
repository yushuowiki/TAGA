# MLP simpler version
import torch
import os
import io
import numpy as np
import random
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
import PIL.Image


from clitox import load_csv, GraphConvNet,GraphClassifier
from neuralfingerprint.mol_graph import degrees


# Plotting parameters
num_figs_per_fp = 3
figsize = (300, 300)
highlight_color = (30.0/255.0, 100.0/255.0, 255.0/255.0)  # A nice light blue.

# Set random seed for reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def remove_duplicates(values, key_lambda):
    output = []
    seen = set()
    for value in values:
        # If value has not been encountered yet, add it to both list and set.
        cur_key = key_lambda(value)
        if cur_key not in seen:
            output.append(value)
            seen.add(cur_key)
    return output

def draw_molecule_with_highlights(filename, smiles, highlight_atoms,
                                  highlight_color=(30/255, 100/255, 255/255),
                                  size=(450, 450)):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")

    highlight_atoms = [int(i) for i in highlight_atoms]

    # Automatically select drawer (.pdf/.svg use SVG vector, .png use Cairo)
    if filename.lower().endswith(('.pdf', '.svg')):
        drawer = Draw.rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
    else:
        drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])

    dos = drawer.drawOptions()

    # Force black and white coloring of atoms
    dos.useBWAtomPalette()                    
    dos.symbolColour = (0, 0, 0)              
    # ========================================

    dos.backgroundColour = (1, 1, 1)          # White background
    dos.bondLineWidth = 3
    dos.additionalAtomLabelPadding = 0.18
    dos.fixedBondLength = 40

    # Highlight connected bonds too
    highlight_bonds = []
    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()
        a2 = bond.GetEndAtomIdx()
        if a1 in highlight_atoms or a2 in highlight_atoms:
            highlight_bonds.append(bond.GetIdx())

    drawer.DrawMolecule(
        mol,
        highlightAtoms=highlight_atoms,
        highlightBonds=highlight_bonds,
        highlightAtomColors={i: highlight_color for i in highlight_atoms},
        highlightBondColors={i: highlight_color for i in highlight_bonds}
    )
    drawer.FinishDrawing()

    # Write SVG directly to file, PDF also via SVG (perfect vector)
    if filename.lower().endswith(('.pdf', '.svg')):
        with open(filename, "w") as f:
            f.write(drawer.GetDrawingText())
    else:
        drawer.WriteDrawingText(filename)   # Cairo generates PNG, SVG backend also creates content

    print(f"Saved: {filename}")


def plot(trained_model_weights_path, trained_classifier_weights_path, config, smiles, labels):
    """
    Modified visualization function for GraphConvNet model
    
    Args:
        trained_model_weights_path: Pretrained GraphConvNet model weights path
        trained_classifier_weights_path: Pretrained classifier weights path
        config: Configuration object containing model hyperparameters
    """
    print("Loading training data...")

    # Create model instances and load pretrained weights
    model = GraphConvNet(config)
    classifier = GraphClassifier(config)
    
    print("Loading pretrained model weights...")
    model.load_state_dict(torch.load(trained_model_weights_path, map_location='cpu'), strict=False)
    classifier.load_state_dict(torch.load(trained_classifier_weights_path, map_location='cpu'))
    
    model.eval()
    classifier.eval()
    
    print("Extracting atom activations...")
    
    # Convert all training data to graph representation
    array_rep = model.array_rep_from_smiles(tuple(smiles), device='cpu')
    atom_features = array_rep['atom_features']
    bond_features = array_rep['bond_features']
    
    # Store activation values for each layer
    atom_activations = {}
    
    # Perform forward propagation and record activation values for each layer
    num_layers = len(model.num_hidden_features)
    for layer in range(num_layers):
        # Calculate fingerprint output for current layer (before softmax)
        pre_softmax, atom_outputs = model.write_to_fingerprint(atom_features, layer)
        
        # Save activation values (convert to numpy array for further processing)
        atom_activations[layer] = atom_outputs.detach().numpy()
        
        # Update atom_features for next layer
        atom_features = model.update_layer(atom_features, bond_features, array_rep, layer)
    
    # Process final layer
    pre_softmax, atom_outputs = model.write_to_fingerprint(atom_features, num_layers)
    atom_activations[num_layers] = atom_outputs.detach().numpy()
    
    if not os.path.exists('clitox_figures'): 
        os.makedirs('clitox_figures')
    
    # Create special directory
    special_dir = 'clitox_figures/special'
    if not os.path.exists(special_dir):
        os.makedirs(special_dir)

    # Build mapping from atom to molecule
    parent_molecule_dict = {}
    for mol_ix, atom_ixs in enumerate(array_rep['atom_list']):
        for atom_ix in atom_ixs:
            parent_molecule_dict[atom_ix] = mol_ix

    # Build neighbor atom list
    atom_neighbor_list = []
    for degree in degrees:
        if ('atom_neighbors', degree) in array_rep:
            neighbors = array_rep[('atom_neighbors', degree)]
            atom_neighbor_list.extend([list(n) for n in neighbors.numpy()])

    def get_neighborhood_ixs(cur_atom_ix, radius):
        """
        Recursive function to get all atom indices within specified radius
        """
        if radius == 0:
            return {cur_atom_ix}
        else:
            cur_set = {cur_atom_ix}
            if cur_atom_ix < len(atom_neighbor_list):
                for n_ix in atom_neighbor_list[cur_atom_ix]:
                    cur_set.update(get_neighborhood_ixs(n_ix, radius-1))
            return cur_set

    # Get weights from the final layer of the classifier
    last_layer_weights = None
    for name, param in classifier.named_parameters():
        if '0.weight' in name:  # Weights of the first linear layer
            last_layer_weights = param.detach().numpy()
            break

    if last_layer_weights is None:
        print("Could not find last layer weights")
        return

    # Define substructure SMARTS templates (for matching toxicity-related groups)
    smarts_sulfur_aromatic = '[S]-[c]'  # Contains sulfur atom connected to aromatic ring
    smarts_fused_aromatic = '[aR2]'#'[a]1[a][a][a][a][a]1'  # Fused aromatic rings (using naphthalene as example)
    pattern_sulfur = Chem.MolFromSmarts(smarts_sulfur_aromatic)
    pattern_fused = Chem.MolFromSmarts(smarts_fused_aromatic)

    # Add new structural patterns
    Sulfonate = '[#6]-S(=O)(=O)-[O,N]'
    Aromatic_amine = '[#6]-[N;H2]'
    Nitroarene = '[#6]-[N+](=O)[O-]'#'[#6]-[N;H1]-[#6]'
    Usaturated_bond = '[#6]=[#6]-[C&$(C=O)]'
    pattern_sulfur = Chem.MolFromSmarts(Sulfonate)
    pattern_fused = Chem.MolFromSmarts(Aromatic_amine)
    pattern_sulfur = Chem.MolFromSmarts(Nitroarene)
    pattern_fused = Chem.MolFromSmarts(Usaturated_bond)


    # Generate visualizations for each fingerprint dimension
    all_radii = range(config.fp_depth + 1)

    # Count total images and toxic ones
    cnt=0
    tox_cnt=0
    
    for fp_ix in range(min(config.fp_length, last_layer_weights.shape[1])):
        print(f"FP {fp_ix} has linear regression coefficient {last_layer_weights[0, fp_ix]}")
        
        # Collect activation values across all radii, radius = layer
        combined_list = []
        for radius in all_radii:
            if radius in atom_activations:
                fp_activations = atom_activations[radius][:, fp_ix]
                combined_list += [(fp_activation, atom_ix, radius) 
                                for atom_ix, fp_activation in enumerate(fp_activations)]

        # Remove duplicates and sort by activation value
        unique_list = remove_duplicates(combined_list, key_lambda=lambda x: x[0])
        combined_list = sorted(unique_list, key=lambda x: -x[0])

        # Generate multiple visualizations per fingerprint
        for fig_ix in range(min(num_figs_per_fp, len(combined_list))):
            # Find the atom with highest activation
            activation, most_active_atom_ix, cur_radius = combined_list[fig_ix]
            most_activating_mol_ix = parent_molecule_dict[most_active_atom_ix]
            
            mol_smiles = smiles[most_activating_mol_ix]
            mol_label = None
            if labels is not None:
                try:
                    mol_label = labels[most_activating_mol_ix]
                except Exception:
                    mol_label = None

            print(f"--> mol_ix: {most_activating_mol_ix}, smiles: {mol_smiles}, label: {mol_label}")
            
            with open('clitox_figures/fp_visualization_summary.csv', 'a', encoding='utf-8') as outf:
                # Initial write can include header (here simplified as append)
                outf.write(f"{fp_ix},{fig_ix},{mol_smiles},{mol_label},{cur_radius},{activation}\n")

            # Get neighborhood atoms
            highlight_list_our_ixs = get_neighborhood_ixs(most_active_atom_ix, cur_radius)
            
            # Convert to RDKit indices
            highlight_list_rdkit = []
            for our_ix in highlight_list_our_ixs:                            
                if our_ix < len(array_rep['rdkit_ix']):
                    highlight_list_rdkit.append(array_rep['rdkit_ix'][our_ix])

            print(f"radius: {cur_radius}, atom list: {highlight_list_rdkit}, activation: {activation}")
            
            # Check substructure matching (whether to include toxicity-related groups)
            mol = Chem.MolFromSmiles(mol_smiles)
            has_sulfur_pattern = False
            has_fused_pattern = False
            has_sulfonate_pattern = False
            has_aromatic_amine_pattern = False
            has_nitroarene_pattern = False
            has_unsaturated_bond_pattern = False
            
            if mol:
                highlight_set = set(highlight_list_rdkit)
                
                # Check if sulfur connected to aromatic ring pattern is included
                sulfur_matches = mol.GetSubstructMatches(pattern_sulfur)
                for match in sulfur_matches:
                    match_set = set(match)
                    if match_set.issubset(highlight_set):  # Highlight completely contains pattern
                        has_sulfur_pattern = True
                        tox_cnt += 1
                        break
                
                # Check if fused aromatic ring pattern is included
                fused_matches = mol.GetSubstructMatches(pattern_fused)
                for match in fused_matches:
                    match_set = set(match)
                    if match_set.issubset(highlight_set):  # Highlight completely contains pattern
                        has_fused_pattern = True
                        tox_cnt += 1
                        break

                # Check if sulfonate ester pattern is included
                sulfonate_matches = mol.GetSubstructMatches(pattern_sulfur)  # pattern_sulfur now is Sulfonate
                for match in sulfonate_matches:
                    match_set = set(match)
                    if match_set.issubset(highlight_set):  # Highlight completely contains pattern
                        has_sulfonate_pattern = True
                        tox_cnt += 1
                        break
                
                # Check if aromatic amine pattern is included
                aromatic_amine_matches = mol.GetSubstructMatches(pattern_fused)  # pattern_fused now is Aromatic_amine
                for match in aromatic_amine_matches:
                    match_set = set(match)
                    if match_set.issubset(highlight_set):  # Highlight completely contains pattern
                        has_aromatic_amine_pattern = True
                        tox_cnt += 1
                        break
                
                # Check if nitroarene pattern is included
                nitroarene_matches = mol.GetSubstructMatches(pattern_sulfur)  # pattern_sulfur now is Nitroarene
                for match in nitroarene_matches:
                    match_set = set(match)
                    if match_set.issubset(highlight_set):  # Highlight completely contains pattern
                        has_nitroarene_pattern = True
                        tox_cnt += 1
                        break
                
                # Check if unsaturated bond pattern is included
                unsaturated_bond_matches = mol.GetSubstructMatches(pattern_fused)  # pattern_fused now is Usaturated_bond
                for match in unsaturated_bond_matches:
                    match_set = set(match)
                    if match_set.issubset(highlight_set):  # Highlight completely contains pattern
                        has_unsaturated_bond_pattern = True
                        tox_cnt += 1
                        break

            
            # Process filename
            safe_label = str(mol_label) if mol_label is not None else 'None'
            
            # Determine save path based on matched patterns
            if has_sulfur_pattern:
                # Create sulfur pattern folder
                sulfur_dir = 'clitox_figures/sulfur_pattern'
                if not os.path.exists(sulfur_dir):
                    os.makedirs(sulfur_dir)
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            elif has_fused_pattern:
                # Create fused aromatic ring folder
                fused_dir = 'clitox_figures/fused_pattern'
                if not os.path.exists(fused_dir):
                    os.makedirs(fused_dir)
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            elif has_sulfonate_pattern:
                # Create sulfonate ester folder
                sulfonate_dir = 'clitox_figures/sulfonate_pattern'
                if not os.path.exists(sulfonate_dir):
                    os.makedirs(sulfonate_dir)
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            elif has_aromatic_amine_pattern:
                # Create aromatic amine folder
                aromatic_amine_dir = 'clitox_figures/aromatic_amine_pattern'
                if not os.path.exists(aromatic_amine_dir):
                    os.makedirs(aromatic_amine_dir)
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            elif has_nitroarene_pattern:
                # Create nitroarene folder
                nitroarene_dir = 'clitox_figures/nitroarene_pattern'
                if not os.path.exists(nitroarene_dir):
                    os.makedirs(nitroarene_dir)
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            elif has_unsaturated_bond_pattern:
                # Create unsaturated bond folder
                unsaturated_bond_dir = 'clitox_figures/unsaturated_bond_pattern'
                if not os.path.exists(unsaturated_bond_dir):
                    os.makedirs(unsaturated_bond_dir)
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            else:
                # Default folder
                filename = f"your_path/radius_{cur_radius}_motif_{fp_ix}_{safe_label}.svg"
            

            # Draw molecule and highlight active regions
            draw_molecule_with_highlights(
                filename,
                smiles[most_activating_mol_ix],
                highlight_atoms=highlight_list_rdkit)
            
            cnt+=1
        print(f"Total images: {cnt}, Toxic images: {tox_cnt}...")
            
# Function to draw indices
def draw_molecule_with_atom_indices(smiles, filename="with_indices.png", size=(450, 450)):
    """Function specifically for viewing atom indices, labeling each atom with its number"""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError("Invalid SMILES")
    
    # Add labels to each atom: just its index
    for atom in mol.GetAtoms():
        atom.SetProp("atomLabel", str(atom.GetIdx()))  # Use index as label directly
    
    drawer = Draw.rdMolDraw2D.MolDraw2DCairo(size[0], size[1])
    dos = drawer.drawOptions()
    dos.useBWAtomPalette()                    # Force black and white
    dos.symbolColour = (0, 0, 0)
    # dos.atomLabels = {i: str(i) for i in range(mol.GetNumAtoms())}  # Double insurance
    dos.additionalAtomLabelPadding = 0.2      # Numbers away from atoms for clarity
    dos.bondLineWidth = 2
    
    Draw.rdMolDraw2D.PrepareAndDrawMolecule(drawer, mol)
    drawer.FinishDrawing()
    
    with open(filename, 'wb') as f:
        f.write(drawer.GetDrawingText())
    
    print(f"Saved image with atom indices: {filename}")
    print("The numbers next to each atom on the image are the indices you need to fill in highlight_atoms!")

if __name__ == "__main__":
    set_seed(42)

    # Simulate configuration parameters
    class Config:
        def __init__(self):
            self.init_scale = 0.018315
            self.h1_size = 100
            self.fp_length = 32
            self.L2_reg = 0.135335
            self.lr = 0.00006
            self.max_atoms = 110
            self.conv_width = 200 # [5, 20]
            self.fp_length = 32 # [16, 128]
            self.fp_depth = 3 # [2,3]
    
    config = Config()
    
    # Set device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load all training data
    print("load single cls test data...")
    # bio 
    all_smiles, all_labels = load_csv("your_data_path/clintox_test.csv")

    label_1_smiles = [s for s,l in zip(all_smiles, all_labels) if l == 1]
    label_1_labels = [1] * len(label_1_smiles)
    label_0_smiles = [s for s,l in zip(all_smiles, all_labels) if l == 0]
    label_0_labels = [0] * len(label_0_smiles)

    selected_smiles = label_1_smiles + label_0_smiles
    selected_labels = label_1_labels + label_0_labels

    trained_model_weights_path = "your_model_weight_path/graphconv_best_clitox.pt"
    trained_classifier_weights_path = "your_head_weight_path/classifier_besta_clitox.pt"
    plot(trained_model_weights_path, trained_classifier_weights_path, config, selected_smiles, selected_labels)

