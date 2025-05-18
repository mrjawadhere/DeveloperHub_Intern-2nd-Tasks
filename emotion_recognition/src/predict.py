import argparse
import torch
import numpy as np
from transformers import BertTokenizerFast, BertForSequenceClassification

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--text', required=True)
    args = parser.parse_args()
    tokenizer = BertTokenizerFast.from_pretrained(args.model_dir)
    model = BertForSequenceClassification.from_pretrained(args.model_dir)
    enc = tokenizer(args.text, return_tensors='pt', truncation=True, padding='max_length', max_length=128)
    with torch.no_grad():
        logits = model(**enc).logits
    probs = torch.sigmoid(logits)[0].cpu().numpy()
    preds = np.where(probs>0.5)[0]
    print('Text:', args.text)
    print('Emotions:', preds.tolist())
    print('Scores:', probs[preds])

if __name__ == '__main__':
    main()
