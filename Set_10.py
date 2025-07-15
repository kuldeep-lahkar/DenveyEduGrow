import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np

# 1. Load students.csv using pandas
df = pd.read_csv("/content/SET_10.csv")

# 2. Handle missing values by replacing them with the mean of the column
df.fillna(df.mean(numeric_only=True), inplace=True)

# 3. Check for and remove any duplicate records
df.drop_duplicates(inplace=True)

# 4. Apply min-max normalization on the numerical columns
numerical_cols = df.select_dtypes(include=['number']).columns
scaler = MinMaxScaler()
df[numerical_cols] = scaler.fit_transform(df[numerical_cols])

# 5. Create a new column Math_Grade by discretizing Math scores
def get_grade(score):
    if score < 0.5:
        return 'F'
    elif score < 0.6:
        return 'D'
    elif score < 0.7:
        return 'C'
    elif score < 0.8:
        return 'B'
    else:
        return 'A'

df['Math_Grade'] = df['Math'].apply(get_grade)

# 6. Smooth noisy data in Math using binning (equal-width binning with 4 bins)
# Create bins
df['Math_Binned'] = pd.cut(df['Math'], bins=4)

# Replace with bin mean (smoothing)
bin_means = df.groupby('Math_Binned')['Math'].transform('mean')
df['Math_Smoothed'] = bin_means

# Display the cleaned DataFrame
print(df.head())
