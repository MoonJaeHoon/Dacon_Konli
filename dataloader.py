from torch.utils.data import Dataset, DataLoader
import torch
import re
import os
import pandas as pd
from augmentation import random_masking_df
import numpy as np

# Dataset 구성.
class NLI_Dataset(Dataset):
    def __init__(self, tokenized_dataset, label):
        self.tokenized_dataset = tokenized_dataset
        self.label = label

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.tokenized_dataset.items()}
        item['label'] = torch.tensor(self.label[idx])
        return item

    def __len__(self):
        return len(self.label)
        
def remove_special_char(args, input_dataset):
    dataset = input_dataset.copy()
    pattern = re.compile(r"[\'\"`%~]")
    dataset['premise'] = dataset['premise'].apply(lambda x: re.sub(pattern,'',x))
    dataset['hypothesis'] = dataset['hypothesis'].apply(lambda x: re.sub(pattern,'',x))
    return dataset

def preprocessing_dataset(args, dataset):

    if 'answer' not in dataset.label: # Train Data일 때에만
        dataset = dataset.loc[dataset.label.isnull()==False,:] 
        dataset = dataset.drop_duplicates(['premise','hypothesis','label']) 

        # Label Encoding
        label_to_num_dict = {'entailment':0,'contradiction':1,'neutral':2,}
        dataset['label'] = dataset.label.map(label_to_num_dict)
    dataset = dataset.reset_index(drop=True)
    return dataset


def load_data(args, dataset_dir):
    print("===================loading data=====================")
    # load dataset
    dataset = pd.read_csv(dataset_dir)
    
    # preprecessing dataset
    dataset = preprocessing_dataset(args, dataset)

    return dataset

def load_data_from_file_lst(args, tokenizer, file_lst):
    print("===========loading data from file_lst=============")
    # load dataset
    dataset = pd.DataFrame(columns=['index','premise','hypothesis','label'])
    for file_name in file_lst:
        print()
        print(f"=========load data (name) : {file_name}===========")

        file_path = os.path.join('./data',file_name)
        concat_dataset = pd.read_csv(file_path)

        if file_name in ['klue_extra_data_sent_perm.csv',
                         'klue_extra_data_rotated_premise.csv', 'klue_extra_data_rotated_hypothesis.csv', 'klue_extra_data_rotated_both.csv', 
                         'klue_extra_data_pororo_en_premise.csv', 'klue_extra_data_pororo_en_hypothesis.csv','klue_extra_data_pororo_en_both.csv',
                         ]:
            concat_dataset = random_masking_df(concat_dataset, tokenizer, threshold=1.0, random_state=42, shuffle=False)
            # concat_dataset = random_masking_col(concat_dataset, tokenizer, col='hypothesis', random_state=64, shuffle=False)
            # concat_dataset = overlap_masking_df(concat_dataset, tokenizer, threshold=1.0, random_state=42)
            pass
        dataset = pd.concat([dataset,concat_dataset],axis=0)

    # preprecessing dataset
    dataset = preprocessing_dataset(args, dataset)
    dataset = dataset.reset_index(drop=True)

    return dataset

# bert input을 위한 tokenizing.
def tokenized_dataset(args, dataset, tokenizer):
    lst_premise = dataset['premise'].tolist()
    lst_hypothesis = dataset['hypothesis'].tolist()

    tokenized_sentences = tokenizer(
        lst_premise,
        lst_hypothesis,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=args.seq_max_len,
        add_special_tokens=True
    )

    return tokenized_sentences

def add_data_from_file_lst(args, tokenizer, dataset, label, file_lst):
    print("===================Adding data from file_lst=====================")
    # load datasets from file_lst
    concat_dataset = load_data_from_file_lst(args, tokenizer, file_lst)
    concat_label = concat_dataset['label'].values

    dataset = pd.concat([dataset, concat_dataset],axis=0)
    label = np.hstack([label, concat_label])
    
    # preprecessing dataset
    dataset = dataset.reset_index(drop=True)

    return dataset, label

def get_trainLoader(args, train_data, valid_data, train_label, valid_label, tokenizer):

    tokenized_train = tokenized_dataset(args, train_data, tokenizer)
    tokenized_valid = tokenized_dataset(args, valid_data, tokenizer)

    # make dataset for pytorch.
    NLI_train_dataset = NLI_Dataset(tokenized_train, train_label)
    NLI_valid_dataset = NLI_Dataset(tokenized_valid, valid_label)

    trainloader = DataLoader(NLI_train_dataset,
                             batch_size=args.batch_size,
                             shuffle=True,
                             num_workers=args.num_workers,
                             )

    validloader = DataLoader(NLI_valid_dataset,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=args.num_workers,
                             )

    return trainloader, validloader

