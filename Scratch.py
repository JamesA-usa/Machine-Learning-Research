import pandas as pd


df = pd.read_csv(r"Top_1k_anomalies_miniLM_150kTraining.csv")
counts_all = df["InitiatingProcessAccountName"].value_counts()

print("")
print("Unique Accounts")
print(counts_all)
print("")

# Keep root and system values while renaming unique user values to user.
df['InitiatingProcessAccountName'] = df['InitiatingProcessAccountName'].apply(
    lambda x: x if pd.isna(x) or str(x).lower() in ['root', 'system'] else 'user'
)