import pandas as pd

df = pd.read_csv("MAIN.csv", sep=";")
print("\n\n".join(list(df.columns)))