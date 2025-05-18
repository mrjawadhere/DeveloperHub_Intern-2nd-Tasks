import argparse
import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizerFast, BertForSequenceClassification, Trainer, TrainingArguments

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def load_data(data_dir):
    df_train = pd.read_csv(f'{data_dir}/train.csv')
    df_val = pd.read_csv(f'{data_dir}/validation.csv')
    return (df_train['text'].tolist(), df_train.drop(columns=['text']).values,
            df_val['text'].tolist(), df_val.drop(columns=['text']).values)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='bert-base-uncased')
    parser.add_argument('--output_dir', required=True)
    args = parser.parse_args()

    from src.data_preprocess import preprocess_and_save
    preprocess_and_save('data')

    tokenizer = BertTokenizerFast.from_pretrained(args.model_name)
    X_train, y_train, X_val, y_val = load_data('data')
    train_ds = EmotionDataset(X_train, y_train, tokenizer)
    val_ds = EmotionDataset(X_val, y_val, tokenizer)

    model = BertForSequenceClassification.from_pretrained(
        args.model_name, num_labels=28, problem_type='multi_label_classification')

    training_args = TrainingArguments(
        output_dir=args.output_dir, evaluation_strategy='epoch',
        per_device_train_batch_size=16, per_device_eval_batch_size=32,
        num_train_epochs=3, logging_dir='./logs', logging_steps=100
    )

    def compute_metrics(p):
        from sklearn.metrics import f1_score, hamming_loss
        logits, labels = p.predictions, p.label_ids
        probs = torch.sigmoid(torch.tensor(logits)).numpy()
        preds = (probs > 0.5).astype(int)
        return {'hamming_loss': hamming_loss(labels, preds),
                'f1_micro': f1_score(labels, preds, average='micro')}

    trainer = Trainer(model=model, args=training_args,
                      train_dataset=train_ds, eval_dataset=val_ds,
                      compute_metrics=compute_metrics)
    trainer.train()
    trainer.save_model(args.output_dir)

if __name__ == '__main__':
    main()
