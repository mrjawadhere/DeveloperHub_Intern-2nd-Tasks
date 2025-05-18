import argparse
import pandas as pd
import torch
from transformers import BertTokenizerFast, BertForSequenceClassification
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import f1_score, hamming_loss

class EmotionDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts, self.labels = texts, labels
        self.tokenizer, self.max_len = tokenizer, max_len

    def __len__(self): return len(self.texts)
    def __getitem__(self, idx):
        enc = self.tokenizer(self.texts[idx], truncation=True, padding='max_length', max_length=self.max_len, return_tensors='pt')
        item = {k: v.squeeze(0) for k,v in enc.items()}
        item['labels'] = torch.tensor(self.labels[idx], dtype=torch.float)
        return item

def load_test(data_dir):
    df_test = pd.read_csv(f'{data_dir}/test.csv')
    return df_test['text'].tolist(), df_test.drop(columns=['text']).values

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    X_test, y_test = load_test('data')
    test_ds = EmotionDataset(X_test, y_test, tokenizer)
    loader = DataLoader(test_ds, batch_size=32)
    preds, labels = [], []
    model.eval()
    with torch.no_grad():
        for batch in loader:
            inputs = {k:v for k,v in batch.items() if k!='labels'}
            logits = model(**inputs).logits
            probs = torch.sigmoid(logits).cpu().numpy()
            preds.append((probs>0.5).astype(int))
            labels.append(batch['labels'].cpu().numpy())
    preds = np.vstack(preds); labels = np.vstack(labels)
    print(f'Hamming Loss: {hamming_loss(labels, preds):.4f}')
    print(f'Micro F1: {f1_score(labels, preds, average="micro"):.4f}')

if __name__ == '__main__':
    main()
