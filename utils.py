import numpy as np
import os
import random
import torch

def wrong_batch_for_wandb(tokenizer,
                          wrong_sample_index,
                          input_ids,
                          valid_labels,
                          valid_predict,
                          valid_output,
                          ):
    num_to_label_dict = {0:'entailment',1:'contradiction',2:'neutral',}

    wrong_sample_index = np.where(valid_labels!=valid_predict)[0]
    wrong_sample_text = [tokenizer.decode(element, skip_special_tokens=False) for element in input_ids[wrong_sample_index]]
    wrong_sample_label = [num_to_label_dict[lab] for lab in list(valid_labels[wrong_sample_index])]
    wrong_sample_pred = [num_to_label_dict[pred] for pred in list(valid_predict[wrong_sample_index])]
    wrong_sample_output = valid_output[wrong_sample_index].tolist()

    entailment_prob, contradiction_prob, neutral_prob = [], [], []
    for element in wrong_sample_output:
        entailment_prob.append(element[0])
        contradiction_prob.append(element[1])
        neutral_prob.append(element[2])

    return wrong_sample_text, wrong_sample_label, wrong_sample_pred, entailment_prob, contradiction_prob, neutral_prob

def seed_everything(seed: int = 42, contain_cuda: bool = False):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print(f"Seed set as {seed}")