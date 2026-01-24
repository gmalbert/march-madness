import pandas as pd

# The headers provided (tab-separated)
headers_tab = "Team	Adj OE	Adj DE	Barthag	Record	Wins	Games	eFG	eFG D.	FT Rate	FT Rate D	TOV%	TOV% D	O Reb%	Op OReb%	Raw T	2P %	2P % D.	3P %	3P % D.	Blk %	Blked %	Ast %	Op Ast %	3P Rate	3P Rate D	Adj. T	Avg Hgt.	Eff. Hgt.	Exp.	Year	PAKE	PASE	Talent		FT%	Op. FT%	PPP Off.	PPP Def.	Elite SOS	Team"

# Split by tab and join with comma
headers = headers_tab.split('\t')
# Remove the last duplicate 'Team'
headers = headers[:-1]
header_line = ','.join(headers)

# Read the CSV without headers
df = pd.read_csv('data_files/trank_team_table_data.csv', header=None)

# Add the header
df.columns = headers

# Save with headers
df.to_csv('data_files/barttorvik_ratings.csv', index=False)

print("CSV with headers saved to data_files/barttorvik_ratings.csv")