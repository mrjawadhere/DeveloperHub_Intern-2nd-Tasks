import argparse
import numpy as np
import torch
from torchvision import transforms
from src.train import SimpleCNN
import matplotlib.pyplot as plt

def load_model(model_dir):
    model = SimpleCNN()
    model.load_state_dict(torch.load(f'{model_dir}/model.pth', map_location='cpu'))
    model.eval()
    return model

def predict_dir(model, data_dir):
    probs = {}
    for fn in sorted([f for f in os.listdir(data_dir) if f.endswith('.npy')]):
        x = np.load(os.path.join(data_dir, fn))[:3]
        x = torch.tensor(x).unsqueeze(0)
        with torch.no_grad():
            out = torch.softmax(model(x), dim=1)[0,1].item()
        probs[fn] = out
    return probs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', required=True)
    parser.add_argument('--t1_dir', required=True)
    parser.add_argument('--t2_dir', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    model = load_model(args.model_dir)
    p1 = predict_dir(model, args.t1_dir)
    p2 = predict_dir(model, args.t2_dir)
    diff = {k: p2[k]-p1[k] for k in p1}
    # visualize as bar over filenames
    plt.figure(figsize=(10,4))
    keys = list(diff.keys())
    vals = [diff[k] for k in keys]
    plt.bar(range(len(vals)), vals)
    plt.xticks(range(len(vals)), keys, rotation=90)
    plt.title('Deforestation Change Scores')
    plt.tight_layout()
    plt.savefig(args.output)

if __name__ == '__main__':
    main()
