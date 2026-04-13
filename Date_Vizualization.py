import pandas as pd
import matplotlib.pyplot as plt

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

# Count root, user, and system accounts.
counts = df['InitiatingProcessAccountName'].value_counts()

# Bar chart. Display counts.
plt.figure(figsize=(14, 7))
plt.bar(counts.index, counts.values)
plt.title("Counts of InitiatingProcessAccountName", fontsize=16)
plt.xlabel("InitiatingProcessAccountName", fontsize=14)
plt.ylabel("Count", fontsize=14)
plt.xticks(rotation=45, ha="right", fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()


# Pie chart by percentage.
plt.figure(figsize=(10, 8))
plt.pie(counts.values, labels=counts.index, autopct='%1.1f%%', startangle=90)
plt.title("Distribution of InitiatingProcessAccountName", fontsize=16)
plt.tight_layout()
plt.show()

# Distribution plot of anomalous scores by account type
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

# Filter data for each account type.
root_data = df[df['InitiatingProcessAccountName'] == 'root']['anomaly_score']
user_data = df[df['InitiatingProcessAccountName'] == 'user']['anomaly_score']
system_data = df[df['InitiatingProcessAccountName'] == 'system']['anomaly_score']

# Plot histogram for root.
axes[0].hist(root_data, bins=30, color='red', alpha=0.7, edgecolor='black')
axes[0].set_title(f'Root Anomaly Scores (n={len(root_data)})', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Anomaly Score', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

# Plot histogram for user.
axes[1].hist(user_data, bins=30, color='blue', alpha=0.7, edgecolor='black')
axes[1].set_title(f'User Anomaly Scores (n={len(user_data)})', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Anomaly Score', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

# Plot histogram for system.
axes[2].hist(system_data, bins=30, color='green', alpha=0.7, edgecolor='black')
axes[2].set_title(f'System Anomaly Scores (n={len(system_data)})', fontsize=14, fontweight='bold')
axes[2].set_xlabel('Anomaly Score', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)
axes[2].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.show()


