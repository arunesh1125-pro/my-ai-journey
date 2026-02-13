import pandas as pd

# Create sample CSV file
data = {
    'Date':pd.date_range('2026-01-01', periods=100),
    'Product': ['Laptop','Phone','Tablet','Watch','Headphones'] * 20,
    'Sales': [45000 + i*100 for i in range(100)],
    'Quantity': [2, 3, 1, 4, 5] * 20,
    'Region': ['North','South','East','West','Central'] * 20,
    'Customer Rating': [4.5, 4.2, 3.8, 4.7, 4.1] * 20
}

df = pd.DataFrame(data)
df.to_csv('sales_data.csv', index=False)
print("✅ Created sales_data.csv")
print("\nFirst 10 rows: ")
print(df.head(10))