import os
import argparse

def parse_args(mode="train"):
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", default=42, type=int, help="seed")

    parser.add_argument("--optimizer", default="AdamW", type=str, help="AdamW, Adam, AdamP")
    parser.add_argument("--scheduler", default="linear", type=str, help="linear, cosine, plateau")
    parser.add_argument("--warmup_steps", default=500, type=int, help="n_warmup_steps")
    parser.add_argument("--cycle_mult", default=1.5, type=float, help="Value multiplied to cycle of Cosine Scheduler")
    parser.add_argument("--seq_max_len", default=128, type=int, help="Max Length of Tokenizing Input Sequence")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size")
    parser.add_argument("--epochs", default=15, type=int, help="Number fo Epochs")
    parser.add_argument("--patience", default=5, type=int, help="Values For EarlyStopping in terms of epochs")
    parser.add_argument("--n_splits", default=5, type=int, help="K Values for K-Fold Ensemble")
    parser.add_argument("--lr", default=1e-05, type=float, help="Learning Rate")
    parser.add_argument("--num_workers", default=2, type=int, help="Number of Workers")
    parser.add_argument("--criterion", default="cross", type=str, help="smoothing, focal, cross")
    parser.add_argument("--smoothing", default=0.0, type=float, help="Smoothing Factor in Label Smoothing")
    parser.add_argument("--dp", default=0.0, type=float, help="DropOut Rate")
    parser.add_argument("--model", default="klue_roberta_large", type=str, help="roberta_large, klue_roberta_large, klue_roberta_base_nli, r_roberta, r_klue_roberta, lstm_add_model, gru_add_model, sep_klue_roberta")

    parser.add_argument("--test_file", default="test_data.csv", type=str, help="추론할 데이터 파일 이름")
    parser.add_argument("--train_valid_file_lst", 
                        default=['train_data.csv',], 
                        type=list, 
                        help="학습-검증 분할 로직에 투입될 데이터 파일들의 리스트")
    parser.add_argument("--only_train_file_lst", 
                        default=['klue_extra_data_sent_perm.csv','klue_extra_data_text_infilling_both.csv',], 
                        type=list, 
                        help="검증에는 포함되지 않고, 학습에만 투입될 데이터 파일 리스트")
    parser.add_argument("--only_valid_file_lst", 
                        default=['klue_extra_data.csv',], 
                        type=list, 
                        help="학습에는 사용되지 않고, 검증에만 투입될 데이터 파일 리스트")

    parser.add_argument("--logging_wrong_samples", default=True, type=bool, help="In Validation, Logging Wrong Samples to Wandb")
    parser.add_argument("--aug_Rotated", default=False, type=bool, help="Whether to Rotation Augmentation")
    parser.add_argument("--aug_OverlapMasked", default=False, type=bool, help="Whether to Overlap Masking Augmentation")
    parser.add_argument("--aug_RandomMasked", default=False, type=bool, help="Whether to RandomMasking Augmentation")
    parser.add_argument("--aug_Deletion", default=False, type=bool, help="Whether to Token Deletion Augmentation")
    parser.add_argument("--aug_Filling", default=False, type=bool, help="Whether to Token Filling Augmentation")
    
    parser.add_argument("--project_name", default="wandb_project1", type=str, help="Project Name for Logging on Wandb")
    parser.add_argument("--model_name", default="wandb_project1", type=str, help="Directory Name to Saving Model")

    args = parser.parse_args()

    return args
