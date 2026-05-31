import pandas as pd 
df2=pd.read_csv(r"/home/b_nicewicz/projekt_wdrazanie_ai/ogloszenia_warszawa_detailed_v3.csv")
print(df2.info())
print(df2.price_sqm_zl)
print(df2.price_total_zl)

df2=df2.dropna()

print(df2.info())