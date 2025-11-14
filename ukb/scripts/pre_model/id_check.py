import pandas as pd
df = pd.read_csv("UKB_Protein_BL_withCov.csv")
print(df.columns.tolist()[:20])
