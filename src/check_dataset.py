import pandas as pd

# Path to CSV
# csv_path = "../dataset/train.csv"
csv_path = r"D:\AP - Projects\RCNN Implementation - TorchVision\dataset\train.csv"


# Read CSV
df = pd.read_csv(csv_path)

# Display basic information
print("Number of rows:", len(df))

print("\nColumns:")
print(df.columns.tolist())

print("\nFirst 10 rows:")
print(df.head(10))