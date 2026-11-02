import time
import torch
import torch.nn as nn
import numpy as np
import random
import torch.nn.functional as F
from tqdm import tqdm
import json
import os
from rdkit import Chem
# from torch.utils.tensorboard import SummaryWriter
# from transformers import get_cosine_schedule_with_warmup
from .neuralfingerprint.mol_graph import graph_from_smiles_tuple, degrees, graph_from_smiles
from .neuralfingerprint.features import num_atom_features, num_bond_features
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

# seet seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# sigle classifier to replace llm
class GraphClassifier(nn.Module):
    def __init__(self, config):
        super(GraphClassifier, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(config.fp_length, config.h1_size),  # first hidden layer
            nn.ReLU(),
            nn.BatchNorm1d(config.h1_size),
            nn.Linear(config.h1_size, 2)  # output layer, for binary cls
        )
    
    def forward(self, fp_score):
        return self.classifier(fp_score)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits, target):
        ce_loss = F.cross_entropy(logits, target, reduction='none')
        pt = torch.exp(-ce_loss)  # pt is the probability of the true class
        
        focal_term = ((1 - pt) ** self.gamma) / (1 + pt)  
        
        # apply category weight
        if self.alpha is not None:
            alpha_weight = self.alpha[target]
            loss = alpha_weight * focal_term * ce_loss
        else:
            loss = focal_term * ce_loss
            
        return loss.mean()

# constraint version
def load_data(file_path):
    """
    load and select legal molecules from *_train.jsonl
    """
    train_smiles = []
    labels = []

    # positive and negative analysis
    positive_count = 0
    negative_count = 0
    
    # raw data
    raw_smiles = []
    raw_labels = []
    
    # read all data
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data = json.loads(line)
            smiles = data['smiles']
            label_str = data['conversations'][1]['value']  
            
            # set label
            label = 1 if label_str == 'Yes' else 0
            
            raw_smiles.append(smiles)
            raw_labels.append(label)
    
    print(f"Raw data {file_path} has {len(raw_smiles)} molecules...")
    
    # filter invalid molecules
    valid_smiles = []
    valid_labels = []
    invalid_count = 0
    
    for i, smiles in enumerate(raw_smiles):
        try:
            # construct 2d graphs to filter SMILES.
            molgraph = graph_from_smiles(smiles)

            valid_smiles.append(smiles)
            valid_labels.append(raw_labels[i])
        except:
            # print(f"Invalid SMILES: {smiles}")
            invalid_count += 1
            continue
    
    print(f"Filtered {invalid_count} invalid molecules...")
    
    canonical_smiles_list = []
    remained_smiles = []
    remained_labels = []
    
    for i, smiles in enumerate(valid_smiles):
        try:        
            mol = Chem.MolFromSmiles(smiles)
            # atom_num_dist.append(len(mol.GetAtoms()))  
            canonical_smiles_list.append(Chem.MolToSmiles(Chem.MolFromSmiles(smiles), isomericSmiles=True))
            remained_smiles.append(smiles)
            remained_labels.append(valid_labels[i])
        except:
            # print("not successfully processed smiles: ", smiles)
            invalid_count += 1
            pass
    
    # Limit the number of atoms
    final_smiles = []
    final_labels = []
    
    for i, smiles in enumerate(remained_smiles):
        try:
            if len(Chem.MolFromSmiles(smiles).GetAtoms()) < 101:
                final_smiles.append(smiles)
                final_labels.append(remained_labels[i])
        except:
            invalid_count += 1
            pass
    
    # positive and negative analysis
    for label in final_labels:
        if label == 1:
            positive_count += 1
        else:
            negative_count += 1
    
    train_smiles = final_smiles
    labels = final_labels
    
    return train_smiles, labels

# GraphConvNet is Here

# validate fx for sigle classifier
def validate_model(model, classifier, val_dataloader, device):

    model.eval()
    classifier.eval()
    correct = 0
    total = 0
    val_loss = 0.0
    
    # neg and pos correction accuracy
    positive_correct = 0
    positive_total = 0
    negative_correct = 0
    negative_total = 0

    # bio
    hiv_positive_count = 1215
    hiv_negative_count = 31477

    bace_positive_count = 480
    bace_negative_count = 730

    # pha
    bbbp_positive_count = 1369
    bbbp_negative_count = 262

    positive_count = hiv_positive_count + bace_positive_count # + bbbp_positive_count
    negative_count = hiv_negative_count + bace_negative_count # + bbbp_negative_count

    total_count = positive_count + negative_count
    weights = [(total_count/negative_count, total_count/positive_count)]  
    class_weights = torch.tensor(weights[0], dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    # # 3. Focal loss
    # criterion = FocalLoss(alpha=None, gamma=2.0)
    
    with torch.no_grad():
        for batch_smiles, batch_labels in val_dataloader:
            # batch_labels = batch_labels.float().to(device) BCEloss
            batch_labels = batch_labels.long().to(device)
            
            # forward
            # logits, _ = model(batch_smiles)
            # logits = logits.to(device)
            fp_score, _ = model(batch_smiles) 
            logits = classifier(fp_score)      
            logits = logits.to(device)
            
            # compute loss
            loss = criterion(logits, batch_labels)
            val_loss += loss.item()
            
            # BCE
            # probabilities = torch.sigmoid(logits)
            # predictions = (probabilities > 0.5).float() 

            # cross entropy
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(probabilities, dim=1) 
            
            # acc in all test data
            correct += (predictions == batch_labels).sum().item()
            total += batch_labels.size(0)
            
            positive_mask = (batch_labels == 1)
            negative_mask = (batch_labels == 0)
            
            positive_correct += (predictions[positive_mask] == batch_labels[positive_mask]).sum().item()
            negative_correct += (predictions[negative_mask] == batch_labels[negative_mask]).sum().item()
            
            positive_total += positive_mask.sum().item()
            negative_total += negative_mask.sum().item()
    
    model.train()
    classifier.train()
    
    # compute metrics
    val_acc = correct / total if total > 0 else 0
    avg_val_loss = val_loss / len(val_dataloader)
    
    positive_acc = positive_correct / positive_total if positive_total > 0 else 0
    negative_acc = negative_correct / negative_total if negative_total > 0 else 0
    
    # output detailed statistics
    print(f"  acc: {val_acc:.4f}, Loss: {avg_val_loss:.4f}")
    print(f"  Pos(label=1) acc: {positive_acc:.4f} ({positive_correct}/{positive_total})")
    print(f"  Neg(label=0) acc: {negative_acc:.4f} ({negative_correct}/{negative_total})")
    
    return val_acc, avg_val_loss, positive_acc, negative_acc

# train

# if __name__ == "__main__":
#     set_seed(42)

#     # config
#     class Config:
#         def __init__(self):
#             self.init_scale = 0.018315
#             self.h1_size = 100
#             self.fp_length = 32
#             self.L2_reg = 0.135335
#             self.lr = 0.0001
#             self.max_atoms = 110
    
#     config = Config()
    
#     # set device
#     device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")
    
#     # load data
#     print("load single cls train data...")
#     # hiv_train_smiles, hiv_train_labels = load_data("your_data_path/hiv_train.jsonl")
#     # bace_train_smiles, bace_train_labels = load_data("your_data_path/bace_train.jsonl")
#     # bbbp_train_smiles, bbbp_train_labels = load_data("your_data_path/bbbp_train.jsonl")
#     # bio
#     hiv_train_smiles, hiv_train_labels = load_data("your_data_path/hiv_train.jsonl")
#     bace_train_smiles, bace_train_labels = load_data("your_data_path/bace_train.jsonl")
#     # pha
#     bbbp_train_smiles, bbbp_train_labels = load_data("your_data_path/bbbp_train.jsonl")
    
#     # load eval data
#     print("load single cls eval data...")
#     # hiv_val_smiles, hiv_val_labels = load_data("your_data_path/hiv_valid.jsonl")
#     # bace_val_smiles, bace_val_labels = load_data("your_data_path/bace_valid.jsonl")
#     # bbbp_val_smiles, bbbp_val_labels = load_data("your_data_path/bbbp_valid.jsonl")
#     hiv_val_smiles, hiv_val_labels = load_data("your_data_path/hiv_valid.jsonl")
#     bace_val_smiles, bace_val_labels = load_data("your_data_path/bace_valid.jsonl")
#     bbbp_val_smiles, bbbp_val_labels = load_data("your_data_path/bbbp_valid.jsonl")
    
    
#     model = GraphConvNet(config).to(device)
#     classifier = GraphClassifier(config).to(device) 
    
#     class SMILESDataset(torch.utils.data.Dataset):
#         def __init__(self, smiles_list, labels_list):
#             self.smiles_list = smiles_list
#             self.labels_list = labels_list
        
#         def __len__(self):
#             return len(self.smiles_list)
        
#         def __getitem__(self, idx):
#             return self.smiles_list[idx], self.labels_list[idx]


#     # dataloader
#     train_dataset = SMILESDataset(all_train_smiles, all_train_labels)
#     val_dataset = SMILESDataset(all_val_smiles, all_val_labels)
    
#     train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
#     val_dataloader = DataLoader(val_dataset, batch_size=16, shuffle=False)

#     num_epochs = 100
#     best_val_acc = 0.0
#     best_pos_acc = 0.0
#     patience = 10
#     patience_counter = 0

#     # optimizer = torch.optim.Adam(list(model.parameters()) + list(classifier.parameters()), lr=config.lr, weight_decay=0.0)
#     optimizer = torch.optim.Adam([
#     {'params': model.parameters(), 'lr': config.lr},
#     {'params': classifier.parameters(), 'lr': config.lr}
#     ], weight_decay=0.0)
    

#     hiv_positive_count = 1215
#     hiv_negative_count = 31477
    
#     # BBBP
#     bbbp_positive_count = 1369
#     bbbp_negative_count = 262
    
#     # BACE
#     bace_positive_count = 480
#     bace_negative_count = 730
    
#     # get merge count
#     positive_count = hiv_positive_count + bace_positive_count #+ bbbp_positive_count
#     negative_count = hiv_negative_count + bace_negative_count #+ bbbp_negative_count
    
#     total_count = positive_count + negative_count
#     weights = [(total_count/negative_count, total_count/positive_count)]
#     class_weights = torch.tensor(weights[0], dtype=torch.float32).to(device)
#     criterion = nn.CrossEntropyLoss(weight=class_weights)
    
#     scheduler = ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=2, min_lr=1e-7, verbose=True)
    
#     # TensorBoard writer
#     writer = SummaryWriter(log_dir='./runs/neuralFP_multi_dataset_experiment')
    
#     # model save path
#     # save_path = 'your_project_path/model/graph_encoder/models'
#     save_path = 'your_projrct_path/model/graph_encoder/models'
#     if not os.path.exists(save_path):
#         os.makedirs(save_path, exist_ok=True)
    
#     # ================================= train ==================================
    # model.train()
    # for epoch in range(num_epochs):
    #     total_loss = 0
    #     total = 0

    #     all_train_probs = []
    #     all_train_labels = []
        
    #     # tqdm
    #     pbar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}/{num_epochs}")
    #     for batch_smiles, batch_labels in pbar:
    #         batch_labels = batch_labels.long().to(device)
    #         optimizer.zero_grad()
            
    #         # forward
    #         # logits, _ = model(batch_smiles)
    #         # logits = logits.to(device)

    #         #singel classfier
    #         fp_score, _ = model(batch_smiles)
    #         logits = classifier(fp_score) 
    #         logits = logits.to(device)
            
    #         # compute loss  
    #         loss = criterion(logits, batch_labels)
            
    #         # backward
    #         loss.backward()
    #         optimizer.step()
            
    #         total_loss += loss.item()
            
    #         # compute AUC-ROC
    #         with torch.no_grad():
    #             probabilities = torch.softmax(logits, dim=1)

    #             all_train_probs.append(probabilities[:, 1])  
    #             all_train_labels.append(batch_labels)
                

    #             total += batch_labels.size(0)
            
    #         # updated tqdm bar
    #         if len(all_train_probs) > 1:  
    #             all_probs = torch.cat(all_train_probs, dim=0).cpu().numpy()
    #             all_labels = torch.cat(all_train_labels, dim=0).cpu().numpy()
    #             from sklearn.metrics import roc_auc_score
    #             try:
    #                 train_auc = roc_auc_score(all_labels, all_probs)
    #                 pbar.set_postfix({
    #                     'loss': f'{loss.item():.4f}',
    #                     'auc': f'{train_auc:.4f}'
    #                 })
    #             except:
    #                 pbar.set_postfix({
    #                     'loss': f'{loss.item():.4f}',
    #                     'auc': 'N/A'
    #                 })
    #         else:
    #             pbar.set_postfix({
    #                 'loss': f'{loss.item():.4f}',
    #                 'auc': 'N/A'
    #             })
        
    #     # avg AUC-ROC
    #     avg_loss = total_loss / len(train_dataloader)
        
    #     # final AUC-ROC
    #     all_train_probs = torch.cat(all_train_probs, dim=0).cpu().numpy()
    #     all_train_labels = torch.cat(all_train_labels, dim=0).cpu().numpy()
    #     train_auc = roc_auc_score(all_train_labels, all_train_probs)

        
    #     # eval
    #     val_auc, val_loss = validate_model(model, classifier, val_dataloader, device)
        
    #     scheduler.step(val_auc)
    #     current_lr = optimizer.param_groups[0]['lr']
        
    #     # TensorBoard
    #     writer.add_scalar('Loss/train', avg_loss, epoch)
    #     writer.add_scalar('Loss/val', val_loss, epoch)
    #     writer.add_scalar('AUC_ROC/train', train_auc, epoch)
    #     writer.add_scalar('AUC_ROC/val', val_auc, epoch)
    #     writer.add_scalar('LearningRate', current_lr, epoch)
        
    #     # output
    #     print(f"Epoch {epoch + 1}/{num_epochs}: "
    #         f"Train Loss={avg_loss:.4f}, Train AUC={train_auc:.4f}, "
    #         f"Val Loss={val_loss:.4f}, Val AUC={val_auc:.4f}, "
    #         f"LR={current_lr:.6f}")
        
    #     # save the best model 
    #     if val_auc > best_val_acc:
    #         best_val_acc = val_auc 
    #         patience_counter = 0
    #         torch.save(model.state_dict(), os.path.join(save_path, "graphconv_best_hiv.pt"))
    #         torch.save(classifier.state_dict(), os.path.join(save_path, "classifier_besta_hiv.pt"))
    #         print("Saved the best model...")
    #     else:
    #         patience_counter += 1
    #         if patience_counter >= patience:
    #             print(f"AUC-ROC showed no improvement within the {patience} epoch, triggering the early stop mechanism.")
    #             break
        
    #     # save each 10 epoch（without classifier）
    #     if (epoch + 1) % 5 == 0:
    #         torch.save(model.state_dict(), 
    #                 os.path.join(save_path, f"graphconv_hiv_epoch_{epoch+1}.pt"))
    #         torch.save(classifier.state_dict(),
    #                 os.path.join(save_path, f"classifier_hiv_epoch_{epoch+1}.pt"))
    
    # writer.close()
    
    # test_datasets = {
    #     "hiv": "your_data_path/hiv_test_gimet.jsonl",
    #     # "bace": "your_data_path/bace_test.jsonl",
    #     # "bbbp": "your_data_path/bbbp_test.jsonl"
    # }

    # for dataset_name, test_file in test_datasets.items():
    #     if os.path.exists(test_file):
    #         if dataset_name == "hiv":
    #             test_smiles, test_labels = save_filter_data(test_file)
    #         else:
    #             test_smiles, test_labels = load_data(test_file)
    #         test_dataset = SMILESDataset(test_smiles, test_labels)
    #         test_dataloader = DataLoader(test_dataset, batch_size=16, shuffle=False)
            
    #         # load best model
    #         model.load_state_dict(torch.load(os.path.join(save_path, "graphconv_best_hiv.pt")), strict=False)
    #         classifier.load_state_dict(torch.load(os.path.join(save_path, "classifier_besta_hiv.pt")), strict=False)
    #         test_auc, _ = validate_model(model, classifier, test_dataloader, device)
    #         print(f"{dataset_name}AUC: {test_auc:.4f}")
