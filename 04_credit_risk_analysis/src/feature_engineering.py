import argparse
import pandas as pd

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    # Example features
    df['DebtToIncomeRatio'] = df['MonthlyDebt'] / (df['MonthlyIncome'] + 1e-8)
    df['AgeBin'] = pd.cut(df['Age'], bins=[0, 30, 50, 100], labels=[0,1,2])
    df = pd.get_dummies(df, columns=['AgeBin'], drop_first=True)
    df.to_csv(args.output, index=False)
    print(f'Features saved to {args.output}')

if __name__ == '__main__':
    main()
