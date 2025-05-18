import argparse
import os
import numpy as np
import rasterio
from rasterio.enums import Resampling
from tqdm import tqdm

def process_image(input_path, output_path):
    with rasterio.open(input_path) as src:
        # read all bands, resample to 256x256
        data = src.read(
            out_shape=(src.count, 256, 256),
            resampling=Resampling.bilinear
        )
        # normalize per-band
        data = data.astype('float32')
        for i in range(data.shape[0]):
            band = data[i]
            data[i] = (band - band.min()) / (band.max() - band.min() + 1e-8)
        np.save(output_path, data)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', required=True, help='Raw TIFF directory')
    parser.add_argument('--output_dir', required=True, help='Processed .npy output directory')
    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)
    for fname in tqdm(os.listdir(args.input_dir)):
        if fname.endswith('.tif'):
            in_path = os.path.join(args.input_dir, fname)
            out_path = os.path.join(args.output_dir, fname.replace('.tif', '.npy'))
            process_image(in_path, out_path)

if __name__ == '__main__':
    main()
