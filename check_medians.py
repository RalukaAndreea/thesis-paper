import pandas as pd
import numpy as np

germ = pd.read_csv("IARC_TP53_DB/GermlineDownload_r21.csv")
soma = pd.read_csv("IARC_TP53_DB/TumorVariantDownload_r21-2.csv")

germ_miss = germ[germ["Effect"] == "missense"].copy()
soma_miss = soma[soma["Effect"] == "missense"].copy()

combined = pd.concat([germ_miss, soma_miss], ignore_index=True)
combined = combined.drop_duplicates(subset=["WT_AA", "Mutant_AA", "Codon_number"]).copy()

revel_vals = pd.to_numeric(combined["REVEL"], errors="coerce").dropna()
bayes_vals = pd.to_numeric(combined["BayesDel"], errors="coerce").dropna()

print("REVEL median:", revel_vals.median())
print("REVEL mean:", revel_vals.mean())
print("BAYESDEL median:", bayes_vals.median())
print("BAYESDEL mean:", bayes_vals.mean())
