import pandas as pd
import numpy as np

"""
PANDAS = Panel Data (for working with tabular/spreadsheet data)
Two main structures:
1. Series (1D labeled array)
2. DataFrame (2D labeled array - like Excel spreadsheet)
"""

 # CREATING SERIES

print("="*60)
print("PANDAS SERIES")
print("="*60)

# From list
data = [10,20,30,40,50]
series1 = pd.Series(data)
print("Series from list: ")
print(series1)
print()

# with custom index
series2 = pd.Series(data, index=['a','b','c','d','e'])
print("Series with custom index: ")
print(series2)
print()

# From dictionary
data_dict = {'Mon':100, 'Tue':150, 'Wed':120, 'Thu':180, 'Fri':200}
series3 = pd.Series(data_dict)
print("Series from dictionary: ")
print(series3)
print()

# From Numpy array
np_array = np.random.randint(1, 100, size=5)
series4 = pd.Series(np_array, index=['Jan', 'Feb', 'Mar', 'Apr', 'May'])
print("Series from Numpy: ")
print(series4)
print()

 # ACCESSING ELEMENTS

print("="*60)
print("ACCESSING SERIES ELEMENTS")
print("="*60)

sales = pd.Series([100, 150, 120, 180, 200],
                  index = ['Mon','Tue','Wed','Thu','Fri'])
print("Sales data: ")
print(sales)
print()

# By index label
print("Monday sales: ", sales['Mon'])
print("Wednesday sales: ", sales['Wed'])
print()

# By position (like NumPy)
print("First day: ", sales[0])
print("Last day: ", sales[-1])
print()

# Slicing
print("Mon-Wed: ")
print(sales['Mon':'Wed'])
print()

# Multiple selection
print("Mon and Fri:")
print(sales[['Mon', 'Fri']])
print()

 # OPERATIONS ON SERIES

print("="*60)
print("SERIES OPERATIONS")
print("="*60)

# Arithmetic (vectorized like NumPy!)
print("Sales * 1.1 (10% increase): ")
print(sales*1.1)
print()

# Boolean indexing
print("Days with sales > 140: ")
print(sales[sales > 140])
print()

# Statistical operations
print(f"Total sales: {sales.sum()}")
print(f"Average sales: {sales.mean():.2f}")
print(f"Max sales: {sales.max()}")
print(f"Min sales: {sales.min()}")
print(f"Std deviation: {sales.std():.2f}")
print()

 # SERIES METHODS

print("="*60)
print("USEFUL SERIES METHODS")
print("="*60)

data1 = pd.Series([1, 2, 2, 3, 3, 3, 4, 4, 4, 4])

print("Data: ", data1.values)
print("Unique values: ", data1.unique())
print("Value counts: ")
print(data1.value_counts())
print()

# Describe (summary statistics)
print("Description: ")
print(sales.describe())
print()

 # HANDLING MISSING DATA

print("="*60)
print("MISSING DATA")
print("="*60)

data_with_nan = pd.Series([1, 2, np.nan, 4, np.nan, 6])
print("Data with NaN: ")
print(data_with_nan)
print()

# Check for missing values
print("Is null?")
print(data_with_nan.isnull())
print()

# Drop missing values
print("After dropna(): ")
print(data_with_nan.dropna())
print()

# Fill missing values
print("After fillna(0): ")
print(data_with_nan.fillna(0))
print()

# Forward fill
print("After ffill() [forward fill]: ")
print(data_with_nan.ffill())
print()

 # SERIES ALIGNMENT

print("="*60)
print("SERIES ALIGNMENT (Powerful Feature!)")
print("="*60)

sales_week1 = pd.Series([100,150,120], index=['Mon','Tue','Wed'])
sales_week2 = pd.Series([110,140,130,170], index=['Mon','Tue','Thu','Fri'])

print("Week 1: ")
print(sales_week1)
print("\nWeek 2: ")
print(sales_week2)
print()

#Add (automatically align by index!)
total = sales_week1 + sales_week2
print("Total (week1 + week2): ")
print(total)
print("\n💡 Notice: Missing indices become NaN!")