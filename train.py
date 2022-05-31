import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')
import os

from model import get_tokenizer, get_model
from dataloader import load_data_from_file_lst, add_data_from_file_lst, get_trainLoader
from sklearn.model_selection import StratifiedKFold
import torch
from sklearn.metrics import accuracy_score
from augmentation import random_masking_df
from optimizer import get_optimizer
from scheduler import get_scheduler
from tqdm import tqdm
from collections import defaultdict
from utils import wrong_batch_for_wandb, seed_everything
from args import parse_args
from loss import get_criterion

import wandb
wandb.login()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f'current device : {device}')

def train(args, wandb):
    criterion = get_criterion(args)
    tokenizer = get_tokenizer(args)
    # all_dataset = load_data(args, dataset_dir = f'./data/{args.train_file}')
    all_dataset = load_data_from_file_lst(args, tokenizer, args.train_valid_file_lst)
    all_label = all_dataset['label'].values

    kf = StratifiedKFold(n_splits=args.n_splits, random_state=42, shuffle=True)
    fold_idx = 1
    best_val_acc_list = []
    for train_index, test_index in kf.split(all_dataset, all_label):


        run = wandb.init(project=args.project_name)
        wandb.run.name = f'{args.model_name}/{fold_idx}-fold'
        wandb.config.update(args)

        os.makedirs(f'./models/{args.model_name}/{fold_idx}-fold', exist_ok=True)
        ### Model Select
        model = get_model(args)
        print('===================get model===================')
        model.to(device)

        train_data, valid_data = all_dataset.iloc[train_index], all_dataset.iloc[test_index]
        train_label, valid_label = all_label[train_index], all_label[test_index]
        
        print(f"len(train_label) : {len(train_label)}")
        print(f"len(valid_label) : {len(valid_label)}")
        # 외부 데이터 활용
        # if args.add_klue_data or args.add_nikl_data: 
        #     train_data, train_label = add_extra_data(args, train_data, train_label)
        
        # # Noising Input (Overlap Masking / Random Masking)
        train_data = random_masking_df(train_data, tokenizer, threshold=0.3, shuffle=False)

        # Augmnetation to Data & Concat
        # aug_data, aug_label = augmentation(args, train_data, tokenizer)
        # train_data = pd.concat([train_data, aug_data],axis=0)
        # train_label = np.hstack([train_label, aug_label])

        # only train 추가
        train_data, train_label = add_data_from_file_lst(args, tokenizer, train_data, train_label, args.only_train_file_lst)
        # only valid 추가
        valid_data, valid_label = add_data_from_file_lst(args, tokenizer, valid_data, valid_label, args.only_valid_file_lst)

        print(f"len(train_label) : {len(train_label)}")
        print(f"len(valid_label) : {len(valid_label)}")

        trainloader, validloader = get_trainLoader(args, train_data, valid_data, train_label, valid_label, tokenizer)	
        total_batch_, valid_batch_ = len(trainloader), len(validloader)

        ### Optimizer
        optimizer = get_optimizer(model, args)

        ### Scheduler
        scheduler = get_scheduler(optimizer, args, total_batch_)

        best_val_loss, best_val_acc, = np.inf, 0
        early_stopping_counter = 0

        print(f"---------------------------------- {fold_idx} fold----------------------------------")	
        for i in tqdm(range(1, args.epochs+1)):
            model.train()
            epoch_perform, batch_perform = np.zeros(2), np.zeros(2)	
            print()
            progress_bar = tqdm(enumerate(trainloader), total=len(trainloader), leave=True, position=0,)
            for j, v in progress_bar:
                input_ids, attention_mask, labels = v['input_ids'].to(device), v['attention_mask'].to(device), v['label'].to(device)

                ###########################
                # if 'roberta' in args.model:
                #     token_type_ids = None
                # else:
                #     token_type_ids = v['token_type_ids'].to(device)
                token_type_ids = None
                ###########################

                optimizer.zero_grad()

                output = model(input_ids, attention_mask, token_type_ids) ## label을 안 넣어서 logits값만 출력	
                # print('='*100)
                # print(output)

                loss = criterion(output, labels)
                # print(f'torch.isfinite(loss) : {torch.isfinite(loss)}')
                loss.backward()
                optimizer.step()
                scheduler.step()
                for learning_rate in scheduler.get_lr():
                    wandb.log({"learning_rate": learning_rate})

                predict = output.argmax(dim=-1)
                predict = predict.detach().cpu().numpy()
                labels = labels.detach().cpu().numpy()	
                acc = accuracy_score(labels, predict)

                batch_perform += np.array([loss.item(), acc])
                epoch_perform += np.array([loss.item(), acc])

                if (j + 1) % 50 == 0:
                    print(
                        f"Epoch {i:#04d} #{j + 1:#03d} -- loss: {batch_perform[0] / 50:#.5f}, acc: {batch_perform[1] / 50:#.4f}"
                        )
                    batch_perform = np.zeros(2)
            print()
            print(
                f"Epoch {i:#04d} loss: {epoch_perform[0] / total_batch_:#.5f}, acc: {epoch_perform[1] / total_batch_:#.2f}"
                )
            wandb.log({
                "epoch": i,
                "Train epoch Loss": epoch_perform[0] / total_batch_,
                "Train epoch Acc": epoch_perform[1] / total_batch_}
                )
            
            ###### Validation	
            model.eval()
            valid_perform = np.zeros(2)

            all_valid_predict_lst = []
            all_valid_labels_lst = []

            # 틀린 데이터들을 wandb 기록하기 위함.
            wrong_sample_dict = defaultdict(list)

            with torch.no_grad():
                for v in tqdm(validloader, total=valid_batch_, leave=True, position=0,):
                    input_ids, attention_mask, valid_labels = v['input_ids'].to(device), v['attention_mask'].to(device), v['label'].to(device)

                    # if 'roberta' in args.model:
                    #     token_type_ids = None
                    # else:
                    #     token_type_ids = v['token_type_ids'].to(device)
                    token_type_ids = None

                    valid_output = model(input_ids, attention_mask, token_type_ids)
                    valid_loss = criterion(valid_output, valid_labels)	

                    valid_predict = valid_output.argmax(dim=-1)
                    valid_predict = valid_predict.detach().cpu().numpy()
                    valid_labels = valid_labels.detach().cpu().numpy()	

                    ###########################
                    # valid eval 결과, 틀린 데이터들은 wandb에 Logging
                    if args.logging_wrong_samples:
                        wrong_sample_index = np.where(valid_labels!=valid_predict)[0]
                        if len(wrong_sample_index)>0:
                            wrong_sample_text, wrong_sample_label, wrong_sample_pred, entailment_prob, contradiction_prob, neutral_prob = wrong_batch_for_wandb(tokenizer, wrong_sample_index, input_ids, valid_labels, valid_predict, valid_output)

                            wrong_sample_dict['입력 문장 Pair'] += wrong_sample_text
                            wrong_sample_dict['실제값'] += wrong_sample_label
                            wrong_sample_dict['예측값'] += wrong_sample_pred
                            wrong_sample_dict['entailment_logit'] += entailment_prob
                            wrong_sample_dict['contradiction_logit'] += contradiction_prob
                            wrong_sample_dict['neutral_logit'] += neutral_prob
                    ###########################

                    valid_acc = accuracy_score(valid_labels, valid_predict)	
                    valid_perform += np.array([valid_loss.item(), valid_acc])

                    all_valid_predict_lst += list(valid_predict)
                    all_valid_labels_lst += list(valid_labels)
            
            ###### Model save
            val_total_loss = valid_perform[0] / valid_batch_
            val_total_acc = valid_perform[1] / valid_batch_
            if best_val_loss > val_total_loss:
                best_val_loss = val_total_loss
                best_epoch_loss = i
        
            if val_total_acc > best_val_acc + 1e-03:    #  + 5e-04
                print(f"New best model for val accuracy : {val_total_acc:#.4f}! saving the best model..")
                torch.save(model.state_dict(), f"./models/{args.model_name}/{fold_idx}-fold/best.pt")
                
                # 참고 : Model 추가 재학습을 위한 모델을 저장하는 코드
                # https://tutorials.pytorch.kr/beginner/saving_loading_models.html#checkpoint

                best_val_acc = val_total_acc
                best_epoch_acc = i
                early_stopping_counter = 0

                ### Confusion Matrix
                class_names = ['entailment','contradiction','neutral'] # (0,1,2)
                # https://wandb.ai/wandb/plots/reports/Confusion-Matrix--VmlldzozMDg1NTM
                wandb.log({f"{i}th_epoch_conf_mat" : wandb.plot.confusion_matrix(probs=None,
                            y_true=all_valid_labels_lst, preds=all_valid_predict_lst,
                            class_names=class_names)})
                
                if args.logging_wrong_samples and val_total_acc > 0.91:
                    ########### Logging Wrong Samples ##########
                    # Save Wrong DataFrame
                    wrong_sample_df = pd.DataFrame(wrong_sample_dict)
                    wrong_sample_df.to_csv(f"./models/{args.model_name}/{fold_idx}-fold/wrong_df.csv",index=False)
                    print('='*15,f'{fold_idx}-Fold Wrong DataFrame Saved','='*15)
                    # Loggin Wandb
                    text_table = wandb.Table(data = wrong_sample_df)
                    run.log({f"{fold_idx}th_fold_wrong_samples" : text_table})
                    ###########################
            
            else: # best보다 score가 안 좋을 때, early stopping check
                early_stopping_counter += 1
                if early_stopping_counter >= args.patience:
                    print(
                        f"EarlyStopping counter: {early_stopping_counter} out of {args.patience}"
                    )
                    print(
                        f">>>> Validation loss: {val_total_loss:#.5f}, Acc: {val_total_acc:#.4f}"
                        )
                    break

            print()
            print(
                f">>>> Validation loss: {val_total_loss:#.5f}, Acc: {val_total_acc:#.4f}"
                )
            print()
            wandb.log({
                    "epoch": i,
                    "Last_Valid Loss": val_total_loss,
                    "Last_Valid Acc": val_total_acc,
                    "Best_Valid Loss": best_val_loss,
                    "Best_Epoch(Loss)": best_epoch_loss,
                    "Best_Valid Acc": best_val_acc,
                    "Best_Epoch(Acc)": best_epoch_acc,
                    }
                )

        best_val_acc_list.append(best_val_acc)
        fold_idx +=1

        gc.collect()
        torch.cuda.empty_cache()

    print('='*50)
    print(f"{args.n_splits}-fold best_val_acc_list : {best_val_acc_list}")
    print('='*15, f'{args.n_splits}-fold Final Score(ACC) : {np.mean(best_val_acc_list)}', '='*15)
    wandb.log({
    f"Total Mean ACC ({args.n_splits}-fold)": np.mean(best_val_acc_list)}
    )


if __name__=='__main__':
    args = parse_args()
    seed_everything(args.seed)
    train(args, wandb)
