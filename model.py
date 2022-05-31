import torch.nn as nn
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModelForSequenceClassification, AutoConfig
from transformers import BertConfig, BertForSequenceClassification, Trainer, TrainingArguments, BertModel, ElectraModel, RobertaModel


class kobert_Classifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=3, dr_rate=0.0):
        super(kobert_Classifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate

        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        # TypeError: dropout(): argument 'input' (position 1) must be Tensor, not tuple
        out = self.bert(input_ids=token_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        out = out[:, 0, :]
        out = self.pooler(out)
        out = torch.nn.functional.tanh(out)  # although BERT uses tanh here, it seems Electra authors used gelu here

        if self.dr_rate:
            out = self.dropout(out)
        
        return self.classifier(out)
    
class koelectra_Classifier(nn.Module):
    def __init__(self, electra, hidden_size=768, num_classes=3, dr_rate=0.0):
        super(koelectra_Classifier, self).__init__()
        self.electra = electra
        self.dr_rate = dr_rate
        
        self.pooler = nn.Linear(hidden_size, hidden_size)
        self.classifier = nn.Linear(hidden_size, num_classes)
        
        torch.nn.init.xavier_uniform_(self.pooler.weight)
        torch.nn.init.xavier_uniform_(self.classifier.weight)
        
        self.dropout = nn.Dropout(p=dr_rate)

    def forward(self, token_ids, attention_mask, segment_ids):
        out = self.electra(input_ids=token_ids, attention_mask=attention_mask, token_type_ids=segment_ids)[0]
        
        out = out[:, 0, :] # take <s> token (equiv. to [CLS])
        out = self.pooler(out)
        out = torch.nn.functional.gelu(out)  # although BERT uses tanh here, it seems Electra authors used gelu here
        if self.dr_rate:
            out = self.dropout(out)

        return self.classifier(out)

class roberta_base_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=768, num_classes=3, dr_rate=0.0):
        super(roberta_base_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate
        
        self.pooler = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.classifier = FCLayer(hidden_size//2, num_classes, self.dr_rate, False)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        out = out[:, 0, :] # take <s> token (equiv. to [CLS])
        out = self.pooler(out)

        return self.classifier(out)

class roberta_large_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=3, dr_rate=0.0):
        super(roberta_large_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate
        
        self.pooler = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.classifier = FCLayer(hidden_size//2, num_classes, self.dr_rate, False)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        out = out[:, 0, :] # take <s> token (equiv. to [CLS])
        out = self.pooler(out)

        return self.classifier(out)


# Relation Extraction R-BERT 아이디어 차용

# https://github.com/monologg/R-BERT/blob/master/model.py#L21
# https://github.com/bcaitech1/p2-klue-arabae/blob/main/model.py
class r_roberta_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=3, dr_rate=0.0):
        super(r_roberta_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate

        self.cls_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.sentence_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.label_classifier = FCLayer(hidden_size//2 * 3, num_classes, self.dr_rate, False)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        sentence_end_position = torch.where(token_ids == 2)[1]
        sent1_end, sent2_end = sentence_end_position[0], sentence_end_position[1]
        
        cls_vector = out[:, 0, :] # take <s> token (equiv. to [CLS])
        prem_vector = out[:,1:sent1_end]              # Get Premise vector
        hypo_vector = out[:,sent1_end+1:sent2_end]    # Get Hypothesis vector

        prem_vector = torch.mean(prem_vector, dim=1) # Average
        hypo_vector = torch.mean(hypo_vector, dim=1)

        
        # Dropout -> tanh -> fc_layer (Share FC layer for premise and hypothesis)
        cls_embedding = self.cls_fc(cls_vector)
        prem_embedding = self.sentence_fc(prem_vector)
        hypo_embedding = self.sentence_fc(hypo_vector)
        
        # Concat -> fc_layer
        concat_embedding = torch.cat([cls_embedding, prem_embedding, hypo_embedding], dim=-1)
        
        return self.label_classifier(concat_embedding)

class sep_roberta_Classifier(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=3, dr_rate=0.0):
        super(sep_roberta_Classifier, self).__init__()
        self.roberta = roberta
        self.dr_rate = dr_rate

        self.cls_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.sentence_fc = FCLayer(hidden_size, hidden_size//2, self.dr_rate)
        self.label_classifier = FCLayer(hidden_size//2 * 3, num_classes, self.dr_rate, False)

    def forward(self, token_ids, attention_mask, segment_ids=None):
        out = self.roberta(input_ids=token_ids, attention_mask=attention_mask)[0]
        
        sentence_end_position = torch.where(token_ids == 2)[1]
        sent1_end, sent2_end = sentence_end_position[0], sentence_end_position[1]
        
        cls_vector = out[:, 0, :] # take <s> token (equiv. to [CLS])
        sep1_vector = out[:,sent1_end, :]    # Get Sep1 token
        sep2_vector = out[:,sent2_end, :]    # Get Sep2 token

        # sep1_vector = torch.mean(sep1_vector, dim=1) # Average
        # sep2_vector = torch.mean(sep2_vector, dim=1)

        
        # Dropout -> tanh -> fc_layer (Share FC layer for premise and hypothesis)
        cls_embedding = self.cls_fc(cls_vector)
        sep1_embedding = self.sentence_fc(sep1_vector)
        sep2_embedding = self.sentence_fc(sep2_vector)
        
        # Concat -> fc_layer
        concat_embedding = torch.cat([cls_embedding, sep1_embedding, sep2_embedding], dim=-1)
        
        return self.label_classifier(concat_embedding)

# LSTM 추가한 모델
# ref : https://github.com/dlrgy22/AI-competition/blob/main/KLUE/code/my_model.py
class lstm_add_model(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=3, dr_rate=0.0):
        super(lstm_add_model, self).__init__()
        self.dr_rate = dr_rate
        self.dropout = nn.Dropout(dr_rate)
        self.roberta = roberta
        self.lstm = nn.LSTM(input_size = 1024, hidden_size = 1024, num_layers = 3, dropout=0.1, bidirectional = True, batch_first = True)
        self.dense_layer = nn.Linear(2048, 3, bias=True)


    def forward(self, input_ids, attention_mask, segment_ids=None):
        # print('=======Traing Forward=======')
        encode_layers = self.roberta(input_ids=input_ids, attention_mask = attention_mask)[0]
        # print(f"encode_layers.shape : {encode_layers.shape}")
        enc_hiddens, (last_hidden, last_cell) = self.lstm(encode_layers)
        # print(f"enc_hiddens.shape : {enc_hiddens.shape}")
        # print(f"last_hidden.shape : {last_hidden.shape}")
        # print(f"last_cell.shape : {last_cell.shape}")
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim = 1)
        # print(f"output_hidden.shape : {output_hidden.shape}")

        output = self.dense_layer(output_hidden)
        # print(f"output.shape : {output.shape}")

        return output

# GRU로 바꿔서도 시도해보기.
class gru_add_model(nn.Module):
    def __init__(self, roberta, hidden_size=1024, num_classes=3, dr_rate=0.0):
        super(gru_add_model, self).__init__()
        self.dr_rate = dr_rate
        self.dropout = nn.Dropout(dr_rate)
        self.roberta = roberta
        self.gru = nn.GRU(input_size = 1024, hidden_size = 1024, num_layers = 3, dropout=0.05, bidirectional = True, batch_first = True)
        self.dense_layer = nn.Linear(2048, 3, bias=True)


    def forward(self, input_ids, attention_mask, segment_ids=None):
        encode_layers = self.roberta(input_ids=input_ids, attention_mask = attention_mask)[0]
        enc_hiddens, last_hidden = self.gru(encode_layers)
        output_hidden = torch.cat((last_hidden[0], last_hidden[1]), dim = 1)

        output = self.dense_layer(output_hidden)

        return output

class FCLayer(nn.Module):
    def __init__(self, input_dim, output_dim, dr_rate=0.0, use_activation=True):
        super(FCLayer, self).__init__()
        self.dr_rate = dr_rate
        self.use_activation = use_activation
        self.dropout = nn.Dropout(dr_rate)
        self.linear = nn.Linear(input_dim, output_dim)
        self.tanh = nn.Tanh()
        
        torch.nn.init.xavier_uniform_(self.linear.weight)

    def forward(self, x):
        if self.use_activation:
            x = self.tanh(x)
        if self.dr_rate:
            x = self.dropout(x)
        return self.linear(x)

def get_tokenizer(args):
    if args.model == 'kobert':
        # tokenizer = KoBertTokenizer.from_pretrained('monologg/kobert')
        tokenizer = AutoTokenizer.from_pretrained("kykim/bert-kor-base")

    elif args.model == 'mbert':
        # tokenizer = AutoTokenizer.from_pretrained("sangrimlee/bert-base-multilingual-cased-korquad")
        tokenizer = AutoTokenizer.from_pretrained('bert-base-multilingual-cased')

    elif args.model == 'koelectra':
        tokenizer = AutoTokenizer.from_pretrained("monologg/koelectra-base-v3-discriminator")

    elif args.model == 'roberta_base':
        # tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-small")
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-base")

    elif args.model == 'roberta_large':
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    elif args.model == 'klue_roberta_small':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-small")

    elif args.model == 'klue_roberta_base':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-base")

    elif args.model == 'klue_roberta_base_nli':
        tokenizer = AutoTokenizer.from_pretrained("Huffon/klue-roberta-base-nli")

    elif args.model == 'klue_roberta_large':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    elif args.model == 'lstm_add_model':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    elif args.model == 'gru_add_model':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    elif args.model == 'r_klue_roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    elif args.model == 'sep_klue_roberta':
        tokenizer = AutoTokenizer.from_pretrained("klue/roberta-large")

    elif args.model == 'r_roberta':
        tokenizer = AutoTokenizer.from_pretrained("xlm-roberta-large")

    else:
        raise NotImplementedError('Tokenizer & Model not available')

    return tokenizer


def get_model(args):
    if args.model == 'kobert':
        # feature_model = BertModel.from_pretrained("monologg/kobert")
        feature_model = BertModel.from_pretrained("kykim/bert-kor-base")
        model = kobert_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'mbert':
        # feature_model = BertModel.from_pretrained("sangrimlee/bert-base-multilingual-cased-korquad")
        feature_model = BertModel.from_pretrained('bert-base-multilingual-cased')
        model = kobert_Classifier(feature_model, dr_rate=args.dp)
    
    elif args.model == 'koelectra':
        feature_model = ElectraModel.from_pretrained("monologg/koelectra-base-v3-discriminator")
        model = koelectra_Classifier(feature_model, dr_rate=args.dp)
    
    elif args.model == 'roberta_base':	# 768
        feature_model = RobertaModel.from_pretrained("xlm-roberta-base", add_pooling_layer=False)
        model = roberta_base_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'roberta_large':	# 1024
        feature_model = RobertaModel.from_pretrained("xlm-roberta-large", add_pooling_layer=False)
        model = roberta_large_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'klue_roberta_small':	# 768
        feature_model = RobertaModel.from_pretrained("klue/roberta-small", add_pooling_layer=False)
        model = roberta_base_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'klue_roberta_base':		# 768
        feature_model = RobertaModel.from_pretrained("klue/roberta-base", add_pooling_layer=False)
        # feature_model = RobertaModel.from_pretrained("Huffon/klue-roberta-base-nli", add_pooling_layer=False)
        model = roberta_base_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'klue_roberta_large':	# 1024
        feature_model = RobertaModel.from_pretrained("klue/roberta-large", add_pooling_layer=False)
        model = roberta_large_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'sep_klue_roberta':	# 1024
        feature_model = RobertaModel.from_pretrained("klue/roberta-large", add_pooling_layer=False)
        model = sep_roberta_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'lstm_add_model':	# 1024
        feature_model = RobertaModel.from_pretrained("klue/roberta-large", add_pooling_layer=False)
        model = lstm_add_model(feature_model, dr_rate=args.dp)

    elif args.model == 'gru_add_model':	# 1024
        feature_model = RobertaModel.from_pretrained("klue/roberta-large", add_pooling_layer=False)
        model = gru_add_model(feature_model, dr_rate=args.dp)

    elif args.model == 'klue_roberta_base_nli':		# 768
        feature_model = RobertaModel.from_pretrained("Huffon/klue-roberta-base-nli", add_pooling_layer=False)
        model = roberta_base_Classifier(feature_model, dr_rate=args.dp)

    elif args.model == 'r_roberta':
        feature_model = RobertaModel.from_pretrained("xlm-roberta-large", add_pooling_layer=False)
        model = r_roberta_Classifier(feature_model, dr_rate=args.dp)
    
    elif args.model == 'r_klue_roberta':
        feature_model = RobertaModel.from_pretrained("klue/roberta-large", add_pooling_layer=False)
        model = r_roberta_Classifier(feature_model, dr_rate=args.dp)

    else:
        raise NotImplementedError('Tokenizer & Model not available')

    return model
