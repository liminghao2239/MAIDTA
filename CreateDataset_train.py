import os
import re
import numpy as np
from torch_geometric.data import InMemoryDataset
from torch_geometric import data as DATA
import torch

class CreateDatasetTrain(InMemoryDataset):
    def __init__(self, root='/', dataset='kiba', drugList=None, protkey=None, y=None, p=None,
                 transform=None, pre_transform=None, smile_graph=None, protein_global=None, drug_smile=None):
        # root is required for save preprocessed data, default is '/tmp'
        super(CreateDatasetTrain, self).__init__(root, transform, pre_transform)

        self.dataset = dataset
        self.root = root
        mol_path = self.processed_paths("mol")
        prot_path = self.processed_paths("prot")
        if os.path.isfile(mol_path) and os.path.isfile(prot_path):
            self.data_mol = torch.load(self.processed_paths("mol"))
            self.data_pro = torch.load(self.processed_paths("prot"))
        else:
            print('Pre-processed data {} not found, doing pre-processing...'.format(self.processed_paths()))
            self.process(drugList=drugList, protkey=protkey, y=y, smile_graph=smile_graph, protein_global=protein_global, p=p, drug_smile=drug_smile)
            #输出看形式 然后save
            # print(type(self.process(drugList=drugList, protkey=protkey, y=y, smile_graph=smile_graph)))
            #self.data, self.slices = torch.load(self.processed_paths())

    def processed_paths(self, filenames="mol"):
        return os.path.join(self.root + self.processed_file_names(filenames)[0])

    def raw_file_names(self):
        return ['some_file_1', 'some_file_2', ...]

    def processed_file_names(self, filenames):
        return [self.dataset + "_" + filenames + '.pt']

    def _process(self):
        pass
        if not os.path.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

    def process(self, drugList, protkey, y, smile_graph, protein_global, p, drug_smile):

        assert(len(drugList) == len(protkey) and len(protkey) == len(y) and len(y) == len(p)) and len(p)==len(drug_smile), "InputFailed: Drug lists, Protein lists and target lists must be the same length"

        data_list_mol = []
        data_list_prot = []

        data_len = len(drugList)
        for i in range(data_len):
            if i % 1000 == 0:
                print("Converting to Graph Dataset : {}/{}".format(i+1, data_len))
            smile = drugList[i]
            target = protkey[i]
            protein = p[i]
            label = y[i]
            drug_smiles = drug_smile[i]

            node_feature, edge_feature, adj_list, edge_type = smile_graph[smile]
            c_size = len(node_feature)



            npy_files_dir = '/tmp/pycharm_project_829/dataset/davis/pdb'
            npy_filename = f"{protein}_pocket.npy"
            file_path = os.path.join(npy_files_dir, npy_filename)

            max_length = 128
            pocket = np.zeros(max_length)

            if os.path.exists(file_path):

                graph_data = np.load(file_path, allow_pickle=True).item()

                pocket_types = graph_data.get('protein_types', [])


                pocket[:len(pocket_types)] = pocket_types[:max_length]  # 截断并填充
            else:
                print(f" {npy_filename} no found")






            # Extract Protein features from Protein graph (Dictionary)
            p_size = len(node_feature)
            # print(type(protein_global))  #dict
            prot_global_feature = protein_global[protein]
            GCNData_mol = DATA.Data(x=torch.Tensor(np.array(node_feature)),
                                edge_attr=torch.Tensor(np.array(edge_feature)),
                                edge_index=torch.LongTensor(np.array(adj_list)).transpose(1, 0),
                                edge_type=torch.Tensor(np.array(edge_type)),
                                y=torch.FloatTensor([label])
                                )
            GCNData_mol.drug_smiles = torch.LongTensor([drug_smiles])
            GCNData_mol.__setitem__('c_size', torch.LongTensor([c_size]))

            ################################################################################################## Waiting for protein representation experiment

            GCNData_prot = DATA.Data(x=torch.Tensor(node_feature),
                                    y=torch.FloatTensor([label]),
                                    x_global=torch.Tensor(np.array(prot_global_feature)))
            GCNData_prot.target = torch.LongTensor([target])
            # GCNData_prot = GCNData_mol.target
            GCNData_prot.pocket = torch.LongTensor(pocket)
            
            GCNData_prot.__setitem__('p_size', torch.LongTensor([p_size]))
            ##################################################################################################


            # Append graph data to data list
            data_list_mol.append(GCNData_mol)
            data_list_prot.append(GCNData_prot)

        if self.pre_filter is not None:
            data_list_mol = [data for data in data_list_mol if self.pre_filter(data)]
            data_list_prot = [data for data in data_list_prot if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list_mol = [self.pre_transform(data) for data in data_list_mol]
            data_list_prot = [self.pre_transform(data) for data in data_list_prot]

        self.data_mol = data_list_mol
        self.data_pro = data_list_prot

        print('\nDataset construction done.\n')
        ############################################################################################### Save only molecule dataset, Waiting for model setting up.
        torch.save(data_list_mol, self.processed_paths("mol"))
        torch.save(data_list_prot, self.processed_paths("prot"))

    def __len__(self):
        return len(self.data_mol)

    def __getitem__(self, idx):
        return self.data_mol[idx], self.data_pro[idx]