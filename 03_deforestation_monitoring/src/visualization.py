import argparse
import matplotlib.pyplot as plt
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--change_map', required=True)
    args = parser.parse_args()
    img = plt.imread(args.change_map)
    plt.figure(figsize=(8,6))
    plt.imshow(img)
    plt.axis('off')
    plt.title('Detected Deforestation Changes')
    plt.show()

if __name__ == '__main__':
    main()
