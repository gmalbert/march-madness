import pandas as pd

def report(path):
    # The updated CSVs contain a duplicate header row; skip the first row
    df = pd.read_csv(path, header=1)
    # ignore empty rows that were padding the CSV
    df = df[df['kenpom'].notna() & (df['kenpom'].astype(str).str.strip()!='')]
    missing = df[df['espn_match'].isna() | (df['espn_match'].astype(str).str.strip()=='')]
    print(path, 'missing count:', len(missing))
    if not missing.empty:
        print(missing[['kenpom','espn_match','suggestions']].to_string(index=False))

report('data_files/kenpom_missing-updated.csv')
print('\n')
report('data_files/bart_missing-updated.csv')
