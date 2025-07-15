import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Load the dataset
df = pd.read_csv("/content/SET-11.csv")
print("Original Dataset:\n", df.head())

# 2. Replace missing values in 'Math' and 'Science' with their column means
df['Math'].fillna(df['Math'].mean(), inplace=True)
df['Science'].fillna(df['Science'].mean(), inplace=True)

# 3. Remove duplicate entries
df.drop_duplicates(inplace=True)

# 4. Normalize numerical columns using Min-Max normalization
num_cols = df.select_dtypes(include=[np.number]).columns
scaler = MinMaxScaler()
df[num_cols] = scaler.fit_transform(df[num_cols])

# 5. Discretize Science marks into categories based on original scale
# Step 1: Reverse normalization on Science to get back actual values
# (Assuming normalized Science is between 0 and 1)
science_min = df['Science'].min()
science_max = df['Science'].max()
df['Science_actual'] = df['Science'] * (science_max - science_min) + science_min

# Step 2: Categorize
def categorize_science(score):
    score = score * 100  # bring it back to 0–100 scale
    if score < 50:
        return 'Poor'
    elif score < 70:
        return 'Average'
    elif score < 90:
        return 'Good'
    else:
        return 'Excellent'

df['Science_Category'] = df['Science'].apply(categorize_science)

# 6. Smooth noisy data using binning (equal-width binning on Science)
# We’ll bin into 4 intervals for simplicity
df['Science_Bin'] = pd.cut(df['Science'], bins=4, labels=['Low', 'Mid-Low', 'Mid-High', 'High'])

# Final dataset preview
print("\nProcessed Dataset:\n", df.head())
