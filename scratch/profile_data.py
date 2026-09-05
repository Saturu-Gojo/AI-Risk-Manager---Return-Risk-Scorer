import pandas as pd
import numpy as np
df = pd.read_csv('train.csv')

print('=== Correlation with returned ===')
num_cols = df.select_dtypes(include='number').columns
corr = df[num_cols].corr()['returned'].sort_values(ascending=False)
print(corr.to_string())

print('\n=== Key data quality issues ===')
print(f'Negative product_price rows: {(df["product_price"] < 0).sum()}')
print(f'Negative num_product_views:  {(df["num_product_views"] < 0).sum()}')
print(f'Negative session_length:     {(df["session_length_minutes"] < 0).sum()}')
print(f'Negative delivery_delay:     {(df["delivery_delay_days"] < 0).sum()}')
print(f'Product rating > 5:          {(df["product_rating"] > 5).sum()}')
print(f'Product rating < 1:          {(df["product_rating"] < 1).sum()}')
print(f'Discount < 0:                {(df["discount_percent"] < 0).sum()}')

# Mutual information for categoricals
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
for c in ['device_type','product_category','shipping_method','payment_method']:
    enc = le.fit_transform(df[c])
    cr = np.corrcoef(enc, df['returned'])[0,1]
    # Return rate by category
    rates = df.groupby(c)['returned'].mean()
    print(f'\n{c} return rates:')
    print(rates.to_string())

print('\n=== Missing occasion_period column ===')
print('occasion_period' in df.columns)
print('Columns:', df.columns.tolist())
