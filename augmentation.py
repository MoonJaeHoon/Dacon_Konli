import pandas as pd
from tqdm import tqdm
import random

## Augmentation to DF
def augmentation(args, input_data, tokenizer):
    aug_dataset = pd.DataFrame(columns=input_data.columns)
    if args.aug_RandomMasked==True:
        concat_dataset = random_masking_col(input_data, tokenizer, col='hypothesis', random_state=64 ,shuffle=False)
        aug_dataset = pd.concat([aug_dataset,concat_dataset],axis=0)
    if args.aug_OverlapMasked==True:
        concat_dataset = overlap_masking_df(input_data, tokenizer, threshold=1.0, random_state=42, shuffle=False)
        aug_dataset = pd.concat([aug_dataset,concat_dataset],axis=0)
    if args.aug_Filling==True:
        concat_dataset = text_infilling_col(input_data, tokenizer, col='hypothesis', threshold=1.0, random_state=108, shuffle=True)
        aug_dataset = pd.concat([aug_dataset,concat_dataset],axis=0)
    if args.aug_Deletion==True:
        concat_dataset = random_token_deletion_col(input_data, tokenizer, col='hypothesis', threshold=1.0, random_state=20, shuffle=True)
        aug_dataset = pd.concat([aug_dataset,concat_dataset],axis=0)
    if args.aug_Rotated==True:
        concat_dataset = rotate_space_df(input_data, tokenizer, threshold=1.0, random_state=96, shuffle=False)
        aug_dataset = pd.concat([aug_dataset,concat_dataset],axis=0)

    aug_label = aug_dataset['label'].values
    print('-'*5,f"aug_dataset.shape : {aug_dataset.shape}",'-'*5,)
    return aug_dataset, aug_label
    
    
## Augmentation (Masking)
# Random Masking to Only One Column
def random_masking_col(input_data, tokenizer, col='premise', random_state=64 ,shuffle=False):
    print('='*15,f'Noising : Random Masking to {col}','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)

    masked_premise_lst = []
    mask_prop_in_onesentence = 0.05

    for hypo in tqdm(origin_data[col].tolist()):
        encoded_ids = tokenizer(hypo)['input_ids']
        seq_len = len(encoded_ids)
        n_masks = min(3, max(1, int(seq_len*mask_prop_in_onesentence)))
        mask_indices = set()

        while len(mask_indices)<n_masks:
            mask_indices.add(random.randrange(1, seq_len-1))

        for ind in list(mask_indices):
            encoded_ids[ind] = 4 # replace to mask 
        d = tokenizer.decode(encoded_ids).replace('[CLS]','').replace('[SEP]','').strip()
        masked_premise_lst.append(d)
    origin_data[col] = masked_premise_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Masking to Premise & Hypo
def random_masking_df(input_data, tokenizer, threshold=0.8, random_state=42, shuffle=False):
    print('='*15,f'Noising : Random Masking to Both Columns','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)
    mask_prop_in_1sentence = 0.05

    masked_new_premise_lst = []
    masked_new_hypo_lst = []
    for origin_premise, origin_hypo in tqdm(zip(origin_data['premise'],origin_data['hypothesis']), total=origin_data.shape[0]):

        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            encoded_premise_ids, encoded_hypo_ids = tokenizer(origin_premise)['input_ids'], tokenizer(origin_hypo)['input_ids']
            premise_seq_len, hypo_seq_len = len(encoded_premise_ids), len(encoded_hypo_ids)
            n_masks_premise, n_masks_hypo = min(3, max(1, int(premise_seq_len*mask_prop_in_1sentence))), min(3, max(1, int(hypo_seq_len*mask_prop_in_1sentence))) # 최소 1개 ~ 최대 3개

            mask_indices = set()
            while len(mask_indices)<n_masks_premise:
                mask_indices.add(random.randrange(1, premise_seq_len-1))
            for ind in list(mask_indices):
                encoded_premise_ids[ind] = 4 # replace to mask 
            new_premise_sentence = tokenizer.decode(encoded_premise_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_premise_lst.append(new_premise_sentence)

            mask_indices = set()
            while len(mask_indices)<n_masks_hypo:
                mask_indices.add(random.randrange(1, hypo_seq_len-1))
            for ind in list(mask_indices):
                encoded_hypo_ids[ind] = 4 # replace to mask 
            new_hypo_sentence = tokenizer.decode(encoded_hypo_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_hypo_lst.append(new_hypo_sentence)

        else:
            masked_new_premise_lst.append(origin_premise)
            masked_new_hypo_lst.append(origin_hypo)
    origin_data['premise'] = masked_new_premise_lst
    origin_data['hypothesis'] = masked_new_hypo_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Overlap Part Masking
def overlap_masking_df(input_data, tokenizer, threshold=0.8, random_state=42, shuffle=False):
    print('='*15,'Noising : Overlap Masking to Both Columns','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)
    mask_prop_in_1row = 0.05
    masked_new_premise_lst = []
    masked_new_hypo_lst = []

    for origin_premise, origin_hypo in tqdm(zip(origin_data['premise'],origin_data['hypothesis']), total=origin_data.shape[0]):
        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            premise_encoded_ids = tokenizer(origin_premise)['input_ids']
            hypo_encoded_ids = tokenizer(origin_hypo)['input_ids']
            seq_len = len(premise_encoded_ids)+len(hypo_encoded_ids)
            overlap_token_ids = [token_ids for token_ids in list(set(premise_encoded_ids)&set(hypo_encoded_ids)) if token_ids not in [0,2]]
            # if len(overlap_token_ids)<=0:
            #     print(overlap_token_ids)

            n_masks = min(len(overlap_token_ids), max(1, int(seq_len*mask_prop_in_1row)))
            mask_indices = random.sample(overlap_token_ids, k=n_masks)
            
            if n_masks==0: # 겹치는 부분 없을 때, Random Masking
                print('=Random Masking=')
                premise_mask_indices = random.sample(premise_encoded_ids,2)
                hypo_mask_indices = random.sample(hypo_encoded_ids,2)
                premise_encoded_ids = [4 if ids in premise_mask_indices else ids for ids in premise_encoded_ids] # replace to mask 
                hypo_encoded_ids = [4 if ids in hypo_mask_indices else ids for ids in hypo_encoded_ids] # replace to mask 

            else: # Overlap Masking
                premise_encoded_ids = [4 if ids in mask_indices else ids for ids in premise_encoded_ids] # replace to mask 
                hypo_encoded_ids = [4 if ids in mask_indices else ids for ids in hypo_encoded_ids] # replace to mask 

            new_premise = tokenizer.decode(premise_encoded_ids).replace('[CLS]','').replace('[SEP]','').strip()
            new_hypo = tokenizer.decode(hypo_encoded_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_premise_lst.append(new_premise)
            masked_new_hypo_lst.append(new_hypo)
        else:
            masked_new_premise_lst.append(origin_premise)
            masked_new_hypo_lst.append(origin_hypo)
    origin_data['premise'] = masked_new_premise_lst
    origin_data['hypothesis'] = masked_new_hypo_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Token Text Infilling to Both Columns
def text_infilling_df(input_data, tokenizer, threshold=0.8, random_state=218, shuffle=False):
    print('='*15,f'Noising : Text Infilling to Both Columns','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)
    mask_prop_in_1sentence = 0.1

    masked_new_premise_lst = []
    masked_new_hypothesis_lst = []
    for origin_premise, origin_hypothesis in tqdm(zip(origin_data['premise'],origin_data['hypothesis']), total=origin_data.shape[0]):

        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            encoded_premise_ids, encoded_hypothesis_ids = tokenizer(origin_premise)['input_ids'], tokenizer(origin_hypothesis)['input_ids']
            premise_seq_len, hypothesis_seq_len = len(encoded_premise_ids), len(encoded_hypothesis_ids)
            n_masks_premise, n_masks_hypothesis = min(3, max(2, int(premise_seq_len*mask_prop_in_1sentence))), min(3, max(2, int(hypothesis_seq_len*mask_prop_in_1sentence))) # 최소 2개 ~ 최대 3개

            mask_indices_1 = random.randrange(1, premise_seq_len-n_masks_premise) # Masking의 시작위치
            mask_indices_2 = mask_indices_1 + n_masks_premise # Masking의 끝나는 위치
            encoded_premise_ids = encoded_premise_ids[:mask_indices_1] + [4] + encoded_premise_ids[mask_indices_2:]
            new_premise_sentence = tokenizer.decode(encoded_premise_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_premise_lst.append(new_premise_sentence)

            mask_indices_1 = random.randrange(1, hypothesis_seq_len-n_masks_hypothesis) # Masking의 시작위치
            mask_indices_2 = mask_indices_1 + n_masks_hypothesis # Masking의 끝나는 위치
            encoded_hypothesis_ids = encoded_hypothesis_ids[:mask_indices_1] + [4] + encoded_hypothesis_ids[mask_indices_2:]
            new_hypothesis_sentence = tokenizer.decode(encoded_hypothesis_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_hypothesis_lst.append(new_hypothesis_sentence)

        else:
            masked_new_premise_lst.append(origin_premise)
            masked_new_hypothesis_lst.append(origin_hypothesis)
    origin_data['premise'] = masked_new_premise_lst
    origin_data['hypothesis'] = masked_new_hypothesis_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Token Text Infilling to 1 Column
def text_infilling_col(input_data, tokenizer, col='premise', threshold=0.8, random_state=108, shuffle=False):
    print('='*15,f'Noising : Text Infilling to {col}','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)
    mask_prop_in_1sentence = 0.1

    masked_new_premise_lst = []
    for origin_premise in tqdm(origin_data[col]):

        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            encoded_premise_ids = tokenizer(origin_premise)['input_ids']
            premise_seq_len = len(encoded_premise_ids)
            n_masks_premise = min(3, max(2, int(premise_seq_len*mask_prop_in_1sentence))) # 최소 2개 ~ 최대 3개

            mask_indices_1 = random.randrange(1, premise_seq_len-n_masks_premise) # Masking의 시작위치
            mask_indices_2 = mask_indices_1 + n_masks_premise # Masking의 끝나는 위치
            encoded_premise_ids = encoded_premise_ids[:mask_indices_1] + [4] + encoded_premise_ids[mask_indices_2:]
            new_premise_sentence = tokenizer.decode(encoded_premise_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_premise_lst.append(new_premise_sentence)

        else:
            masked_new_premise_lst.append(origin_premise)
    origin_data[col] = masked_new_premise_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Token Deletion to Both Columns
def random_token_deletion_df(input_data, tokenizer, threshold=0.8, random_state=22, shuffle=False):
    print('='*15,f'Noising : Token Deletion to Both Columns','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)
    mask_prop_in_1sentence = 0.05

    masked_new_premise_lst = []
    masked_new_hypothesis_lst = []
    for origin_premise, origin_hypothesis in tqdm(zip(origin_data['premise'],origin_data['hypothesis']), total=origin_data.shape[0]):

        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            encoded_premise_ids, encoded_hypothesis_ids = tokenizer(origin_premise)['input_ids'], tokenizer(origin_hypothesis)['input_ids']
            premise_seq_len, hypothesis_seq_len = len(encoded_premise_ids), len(encoded_hypothesis_ids)
            n_masks_premise, n_masks_hypothesis = 1, 1 # 1개

            mask_idx = random.randrange(1, premise_seq_len-1)
            encoded_premise_ids = encoded_premise_ids[:mask_idx] + encoded_premise_ids[mask_idx+1:]
            new_premise_sentence = tokenizer.decode(encoded_premise_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_premise_lst.append(new_premise_sentence)

            mask_idx = random.randrange(1, premise_seq_len-1)
            encoded_hypothesis_ids = encoded_hypothesis_ids[:mask_idx] + encoded_hypothesis_ids[mask_idx+1:]
            new_hypothesis_sentence = tokenizer.decode(encoded_hypothesis_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_hypothesis_lst.append(new_hypothesis_sentence)

        else:
            masked_new_premise_lst.append(origin_premise)
            masked_new_hypothesis_lst.append(origin_hypothesis)
    origin_data['premise'] = masked_new_premise_lst
    origin_data['hypothesis'] = masked_new_hypothesis_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Token Deletion to 1 Column
def random_token_deletion_col(input_data, tokenizer, col='premise', threshold=0.8, random_state=20, shuffle=False):
    print('='*15,f'Noising : Token Deletion to {col}','='*15)
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)
    mask_prop_in_1sentence = 0.05

    masked_new_premise_lst = []
    for origin_premise in tqdm(origin_data[col]):

        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            encoded_premise_ids = tokenizer(origin_premise)['input_ids']
            premise_seq_len = len(encoded_premise_ids)
            n_masks_premise = 1 # 1개

            mask_idx = random.randrange(1, premise_seq_len-1)
            encoded_premise_ids = encoded_premise_ids[:mask_idx] + encoded_premise_ids[mask_idx+1:]
            new_premise_sentence = tokenizer.decode(encoded_premise_ids).replace('[CLS]','').replace('[SEP]','').strip()
            masked_new_premise_lst.append(new_premise_sentence)

        else:
            masked_new_premise_lst.append(origin_premise)
    origin_data[col] = masked_new_premise_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Token Rotation to Both Columns

def rotate_space_df(input_data, tokenizer, threshold=0.8, random_state=96, shuffle=False):
    origin_data = input_data.copy()
    if shuffle==True:
        origin_data = origin_data.sample(len(origin_data), random_state=random_state)

    rotated_new_premise_lst = []
    rotated_new_hypothesis_lst = []
    for origin_premise, origin_hypothesis in tqdm(zip(origin_data['premise'],origin_data['hypothesis']), total=origin_data.shape[0]):

        will_masking_prop = random.random()
        if will_masking_prop <= threshold:
            splitted_premise = origin_premise.strip().split()
            seq_len = max(2, len(splitted_premise))
            slicing_idx = random.randrange(1, seq_len)
            new_premise_sentence = ' '.join(splitted_premise[slicing_idx:] + splitted_premise[:slicing_idx])
            rotated_new_premise_lst.append(new_premise_sentence.strip())

            splitted_hypothesis = origin_hypothesis.strip().split()
            seq_len = max(2, len(splitted_premise))
            slicing_idx = random.randrange(1, seq_len)
            new_hypothesis_sentence = ' '.join(splitted_hypothesis[slicing_idx:] + splitted_hypothesis[:slicing_idx])
            rotated_new_hypothesis_lst.append(new_hypothesis_sentence.strip())

        else:
            rotated_new_premise_lst.append(origin_premise)
            rotated_new_hypothesis_lst.append(origin_hypothesis)
    origin_data['premise'] = rotated_new_premise_lst
    origin_data['hypothesis'] = rotated_new_hypothesis_lst
    origin_data = origin_data.reset_index(drop=True)
    return origin_data

# Random Token Rotation to 1 Column
def rotate_space_col(input_data, tokenizer, col='premise'):
    print('='*15,f'Noising : Token Rotation to {col}','='*15)
    origin_data = input_data.copy()

    rotated_premise_lst = []
    for prem in tqdm(origin_data[col].tolist()):
        splitted_prem = prem.strip().split()
        seq_len = len(splitted_prem)

        slicing_idx = random.randrange(1, seq_len)
        rotated_sentence = ' '.join(splitted_prem[slicing_idx:] + splitted_prem[:slicing_idx])

        rotated_premise_lst.append(rotated_sentence.strip())

    origin_data[col] = rotated_premise_lst
    return origin_data


