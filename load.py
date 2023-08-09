import pandas as pd

df = pd.read_csv("MAIN.csv", sep=";")

if __name__ == "__main__":
    print("\n\n".join(list(df.columns)))