import pandas as pd
from datasets import load_dataset
from sklearn.preprocessing import MultiLabelBinarizer
import os

def preprocess_and_save(output_dir='data'):
    dataset = load_dataset('go_emotions', 'raw')
    mlb = MultiLabelBinarizer(classes=list(range(28)))
    mlb.fit([list(range(28))])
    for split in ['train', 'validation', 'test']:
        ds = dataset[split]
        df = pd.DataFrame({'text': ds['text'], 'labels': ds['labels']})
        df_labels = pd.DataFrame(mlb.transform(df['labels']), columns=[str(i) for i in mlb.classes_])
        df_full = pd.concat([df[['text']], df_labels], axis=1)
        os.makedirs(output_dir, exist_ok=True)
        df_full.to_csv(os.path.join(output_dir, f'{split}.csv'), index=False)

if __name__ == '__main__':
    preprocess_and_save()
