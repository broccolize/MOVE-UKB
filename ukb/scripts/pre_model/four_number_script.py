import pandas as pd

b = pd.read_csv("data/ukb_batch.tsv", sep="\t", nrows=1)
d = pd.read_csv("data/ukb_bloodsample_date_0_0.tsv", sep="\t", nrows=1)
p = pd.read_csv("data/ukb_proteomics.tsv", sep="\t", nrows=1)
c = pd.read_csv("data/ukb_continuous.tsv", sep="\t", nrows=1)

n_batch = b.shape[1]-1
n_date  = d.shape[1]-1
n_prot  = p.shape[1]-1
n_cont  = c.shape[1]-1

print(n_batch, n_date, n_prot, n_cont)
print("sum=", n_batch+n_date+n_prot+n_cont)
