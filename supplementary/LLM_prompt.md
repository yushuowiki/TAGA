### Example ablation prompts

#### BBBP

- BBBP-short

  Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Can this molecule permeate the blood-brain barrier? Answer 'Yes' or 'No'.

- BBBP-origin

  Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on the blood-brain barrier (BBB): “BBB restricts molecular entry into the central nervous system via tight junctions and efflux transporters. Permeable molecules typically have molecular weight <500 Da, logP between 2–4, fewer than 5 hydrogen bond donors, and limited acceptors, avoiding highly polar or charged groups.” Based on this information and the graph, can this molecule effectively permeate the BBB? Answer 'Yes' or 'No'.

- BBBP-rewrite

  Given a 2D molecular graph: <graph>, where tokens encode functional group semantics. The BBB acts as a selective barrier. Molecules that cross it are usually moderately lipophilic, not too large, and have limited polarity. Using this knowledge and the graph, determine whether the molecule is BBB-permeable. Answer 'Yes' or 'No'.

#### QM9

- QM9-short

  Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?

- QM9-origin

  Given a molecule in the form of a 2D graph: <graph>, each token encodes functional group semantics. The LUMO energy reflects the molecule’s ability to accept electrons and is influenced by conjugation and electron-withdrawing groups. Based on the graph, predict the LUMO energy of this molecule.

- QM9-rewrite

  Given a molecular graph: <graph>, where each token represents a functional motif. The LUMO level is related to electron affinity and depends on the electronic structure. Estimate the LUMO energy from the given graph.

### Complete json prompt

### BACE

```json
{"smiles": "O1CC[C@@H](NC(=O)[C@@H](Cc2cc3cc(ccc3nc2N)-c2ccccc2C)C)CC1(C)C", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on BACE1 (β-site amyloid precursor protein cleaving enzyme 1): 'BACE1 is an aspartic-acid protease central to Alzheimer's disease pathogenesis, cleaving amyloid precursor protein (APP) to form β-amyloid peptides that aggregate into plaques. As a bilobal aspartic protease, each lobe contributes a catalytic Asp residue, with an extended active site cleft between lobes. Inhibitors typically form hydrogen bonds with the catalytic aspartates and occupy adjacent hydrophobic pockets for therapeutic intervention in AD.' Based on this information and graph information, can this molecule effectively bind to or inhibit BACE1 in this assay (i.e., should this molecule be labeled as active against BACE1)? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "Yes"}]}
```
#### BBBP

```json
{"smiles": "CCC1(C(=O)NCNC1=O)c2ccccc2", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on the blood-brain barrier (BBB): 'BBB restricts molecular entry into the central nervous system via tight junctions and efflux transporters. Permeable molecules typically have molecular weight <500 Da, logP (octanol-water partition coefficient) between 2-4 for optimal lipophilicity, fewer than 5 hydrogen bond donors, and limited acceptors to facilitate passive diffusion or membrane transport, avoiding polar or charged groups that hinder crossing.' Based on this information and graph information, can this molecule effectively permeate the blood-brain barrier in this assay (i.e., should this molecule be labeled as active for BBB permeability)? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "Yes"}]}
```

#### ClinTox

```json
{"smiles": "C1CN1P(=S)(N2CC2)N3CC3", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on clinical toxicity: 'ClinTox contains drugs that have failed clinical trials for toxicity reasons, and also drugs that have received FDA approval. The dataset helps predict if a molecule will be clinically toxic.' Based on this information and graph information, is this molecule FDA approved? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "Yes"}]}
```

### Cyp450

```json
{"smiles": "Oc1ccc2oc3c(c2c1)CCCC3", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on CYP450 metabolism: 'Cytochrome P450 enzymes are responsible for metabolizing a large number of drugs. Inhibition of these enzymes can lead to drug-drug interactions.' Based on this information and graph information, does this molecule inhibit CYP1A2? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "Yes"}]}
```

### ESOL

```json
{"smiles": "O=C1CCCN1", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on hydration free energy: 'Aqueous solubility (logS) is a key physicochemical property influencing drug absorption and bioavailability. It can be approximated using models like negative LogP -0.01 * (MPt – 25) + 0.5, where LogP reflects hydrophobicity and MPt (melting point) accounts for crystal lattice energy. Factors such as hydrogen bonding, molecular size, and polarity affect solubility, with polar groups enhancing water interactions and hydrophobic moieties reducing it.' Based on this information and graph information, what is the aqueous solubility (logS) of this molecule? Please provide the predicted numerical value."]}, {"from": "gpt", "value": 1.07}]}
```

#### Freesolv

```json
{"smiles": "Cn1cnc2c1c(=O)n(c(=O)n2C)C", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on hydration free energy: 'The free energy of hydration (Δμch) measures the energy change when transferring a solute from vapor to water, crucial for understanding solvation and biomolecular interactions. It can be approximated by ΔG_hyd = ΔG_solv,soln - ΔG_solv,gas + RT ln (10^(-pKa)), incorporating solvation terms and acidity (negative pKa) to predict hydration behavior, influenced by polar interactions, hydrogen bonding, and entropy changes.' Based on this information and graph information, what is the free energy of hydration (Δμch) of this molecule? Please provide the predicted numerical value."]}, {"from": "gpt", "value": -12.64}]}
```

### HIV

```json
{"smiles": "S=c1nnn(C2CCCN2)[nH]1", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on Human immunodeficiency virus (HIV): 'HIV is a retrovirus causing acquired immunodeficiency syndrome (AIDS). Antiretroviral drugs target various stages of the viral lifecycle, including nucleoside reverse transcriptase inhibitors (NRTIs) lacking a 3'-hydroxyl group to prevent DNA chain elongation, non-nucleoside reverse transcriptase inhibitors (NNRTIs) binding allosterically to RT, protease inhibitors forming hydrogen bonds with Asp25/Asp25' in the conserved active site (interacting with residues like Gly27, Asp29, Asp30, Gly48), integrase inhibitors, fusion inhibitors, and CCR5 antagonists. Effective inhibitors often exploit hydrophobic pockets, hydrogen bonding, or metal chelation in viral enzymes.' Based on this information and graph information, can this molecule effectively inhibit HIV replication in this assay (i.e., should this molecule be labeled as active against HIV)? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "No"}]}
```

### Lipo

```json
{"smiles": "O=C1C=COc2cc(OCc3ccccc3)ccc12", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on lipophilicity: 'Lipophilicity, measured as the octanol/water distribution coefficient (logD at pH 7.4), determines membrane permeability and solubility of drugs, affecting pharmacokinetics. At physiological pH, it accounts for ionization states, with higher logD indicating greater lipid affinity via hydrophobic interactions, while polar or charged groups favor the aqueous phase.' Based on this information and graph information, what is the octanol/water distribution coefficient (logD at pH 7.4) of this molecule? Please provide the predicted numerical value."]}, {"from": "gpt", "value": 3.33}]}
```

#### MUV

```json
{"smiles": "CCOC(=O)C1=C(C)NC(=O)NC1C", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on molecular activity: 'MUV (Maximum Unbiased Validation) datasets are derived from PubChem BioAssays, designed for virtual screening benchmarking with challenging decoys.' Based on this information and graph information, is this molecule active in MUV-600 assay? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "No"}]}
```

#### QM9 (LUMO)

```json
{"smiles": "CC1C2CC2NC1C#N", "conversations": [{"from": "human", "value": "Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. What is the lowest unoccupied molecular orbital (LUMO) energy of this molecule?"}, {"from": "gpt", "value": "0.025600"}], "target": "lumo"}
```

#### Sider

```json
{"smiles": "COC1=CC(=CC(=C1OC)OC)CC2=CN=C(N=C2N)N", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on drug side effects: 'SIDER contains information on marketed medicines and their recorded adverse drug reactions. The database links drugs to side effects based on their interaction with various biological targets, including enzymes, receptors, and transporters.' Based on this information and graph information, does this molecule cause Hepatobiliary disorders side effect? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "Yes"}]}
```

#### Tox21

```json
{"smiles": "O=C1CCC(N2C(=O)c3ccccc3C2=O)C(=O)N1", "conversations": [{"from": "human", "value": ["Given a molecule in the form of a 2D graph: <graph>, each token in the graph contains functional group semantic information. Considering the following mechanistic background on toxicity assays: 'Tox21 is a federal collaboration using high-throughput screening to test chemicals for toxicity. The assays measure disruption of biological pathways including nuclear receptors and stress response pathways.' Based on this information and graph information, is this molecule active in NR-AR assay? Please respond with only 'Yes' or 'No'."]}, {"from": "gpt", "value": "No"}]}
```



