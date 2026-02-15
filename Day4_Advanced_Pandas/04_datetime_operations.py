import pandas as pd
import numpy as np
from datetime import datetime, timedelta

print("="*70)
print("DATETIME OPERATIONS IN PANDAS")
print("="*70)

 # CREATING DATETIME DATA

# Create sample sales data
np.random.seed(42)
dates = pd.date_range('2026-01-01', '2026-12-31', freq='D')

sales_data = pd.DataFrame({
    'Date': dates,
    'Sales': np.random.randint(1000, 10000, len(dates)),
    'Orders': np.random.randint(10, 100, len(dates))
})

print("Sales data sample: ")
print(sales_data.head(10))
print()

 # DATETIME COMPONENTS

print("="*70)
print("EXTRACTING DATE COMPONENTS")
print("="*70)

df = sales_data.copy()

# Extract components
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Month_Name'] = df['Date'].dt.month_name()
df['Day'] = df['Date'].dt.day
df['Day_of_Week'] = df['Date'].dt.dayofweek
df['Day_Name'] = df['Date'].dt.day_name()
df['Week'] = df['Date'].dt.isocalendar().week
df['Quarter'] = df['Date'].dt.quarter
df['Is_Weekend'] = df['Day_of_Week'].isin([5, 6])

print("Date components: ")
print(df[['Date', 'Year', 'Month', 'Month_Name',
          'Day_Name', 'Quarter', 'Is_Weekend']].head(10))
print()

# DATE FILTERING

print("="*70)
print("FILTERING BY DATE")
print("="*70)

# Filter specific month
jan_sales = df[df['Month'] == 1]
print(f"Januray sales: {len(jan_sales)} days")
print(f"January total: {jan_sales['Sales'].sum():,}")
print()

# Filter date range
q1 = df[(df['Date'] >= '2026-01-01') & (df['Date'] <= '2026-03-31') ]
print(f"Q1 2026 sales: {len(q1)} days")
print(f"Q1 total: {q1['Sales'].sum():,}")
print()

# Weekend vs weekday
weekend = df[df['Is_Weekend'] == True]
weekday = df[df['Is_Weekend'] == False]

print(f"Average weekend sales: {weekend['Sales'].mean():,.2f}")
print(f"Average weekday sales: {weekday['Sales'].mean():,.2f}")
print()

 # DATE AGGREGATIONS

print("="*70)
print("DATE AGGREGATIONS")
print("="*70)

# Monthly totals
monthly = df.groupby('Month')['Sales'].agg(['sum','mean','max'])
monthly.index = pd.date_range('2026-01', periods=12, freq='M').strftime('%B')
print("Monthly Sales Summary: ")
print(monthly)
print()

# Quarter totals
quarterly = df.groupby('Quarter')['Sales'].sum()
print("Quarterly Sales: ")
print(quarterly)
print()

# Day of week analysis
dow_analysis = df.groupby('Day_Name')['Sales'].mean().round(2)
print(dow_analysis.sort_values(ascending=False))
print()

 # DATE ARTHIMETIC

print("="*70)
print("DATE ARITHMETIC")
print("="*70)

# Calculate days since start
df['Days_Since_Start'] = (df['Date'] - df['Date'].min()).dt.days
print("Days since start: ")
print(df[['Date', 'Days_Since_Start']].head())
print()

# Add/Subtract time
today = pd.Timestamp('2026-02-15')
next_week = today + pd.Timedelta(days=7)
last_month = today - pd.DateOffset(months=1)
next_quarter = today + pd.DateOffset(months=3)

print(f"Today: {today.date()}")
print(f"Next week: {next_week.date()}")
print(f"Last Month: {last_month.date()}")
print(f"Next quarter: {next_quarter.date()}")
print()

# ROLLING STATISTICS (Time Series)

print("="*70)
print("ROLLING STATISTICS (Moving Averages)")
print("="*70)

# 7-day moving average
df['Sales_7day_avg'] = df['Sales'].rolling(window=7).mean()

# 30-day moving average
df['Sales_30day_avg'] = df['Sales'].rolling(window=30).mean()

# 7-day sum
df['Sales_7day_sum'] = df['Sales'].rolling(window=7).sum()

print("Sales with moving averages (first 35 rows): ")
print(df[['Date', 'Sales', 'Sales_7day_avg', 'Sales_30day_avg']].head(35))

 # RESAMPLING (Change time frequency)

print("="*70)
print("RESAMPLING")
print("="*70)

df_indexed = df.set_index('Date')
# Daily -> Weekly
weekly = df_indexed['Sales'].resample('W').sum()
print("Weekly sales (first 5 weeks: )")
print(weekly.head())
print()

# Daily -> Monthly
monthly1 = df_indexed['Sales'].resample('M').agg({
    'sum':'sum',
    'mean': 'mean',
    'max': 'max'
})
print("Monthly sales statistics: ")
print(monthly1)