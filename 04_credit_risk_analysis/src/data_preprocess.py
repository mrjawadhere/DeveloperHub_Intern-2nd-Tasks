import argparse
import pandas as pd
from sklearn.impute import SimpleImputer

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Impute missing values
    imputer = SimpleImputer(strategy='median')
    df[df.columns] = imputer.fit_transform(df)
    df.to_csv(args.output, index=False)
    print(f'Processed data saved to {args.output}')

if __name__ == '__main__':
    main()
