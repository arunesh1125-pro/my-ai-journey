import pandas as pd
import numpy as np
import re

print("="*70)
print("STRING DATA CLEANING")
print("="*70)

# Create messy string data (very common in real world!)
data = {
    'Name': ['  Alice Johnson  ', 'BOB SMITH', 'charlie.brown',
             'DAVID-Lee', 'eve_wilson', 'Frank O\'Brien', '  Grace  Kim  '],
    'Email': ['Alice.Johnson@Gmail.COM', 'bob@HOTMAIL.com',
              'charlie.brown@yahoo.com', 'invalid-email',
              'eve@company.co.in', 'frank.obrien@gmail.com',
              'grace.kim@outlook.com'],
    'Phone': ['9876543210', '+91 98765 43211', '(098) 765-4321',
              '98765-43213', '9876543214 ', '91-9876543215',
              'not-a-phone'],
    'City': ['chennai', 'MUMBAI', 'Delhi ', 'bangalore',
             ' Kolkata ', 'HYDERABAD', 'Pune'],
    'Description': ['Software  Engineer', 'DATA  scientist',
                    'Machine Learning   Engineer', 'AI   researcher',
                    'Full  Stack   Developer', 'DevOps  Engineer',
                    'ML  Engineer']
}

df = pd.DataFrame(data)
print("Messy data: ")
print(df)
print()

# BASIC STRING OPERATIONS

print("="*70)
print("BASIC STRING OPERATIONS")
print("="*70)

df_clean = df.copy()

# Standarize case
df_clean['Name'] = df_clean['Name'].str.strip().str.title()
df_clean['City'] = df_clean['City'].str.strip().str.title()
df_clean['Email'] = df_clean['Email'].str.strip().str.lower()

print("After Standardizing case: ")
print(df_clean[['Name', 'City', 'Email']])
print()

# Remove extra whitespace from description
df_clean['Description'] = df_clean['Description'].str.strip()
df_clean['Description'] = df_clean['Description'].str.replace(r'\s+', ' ', regex=True)
print("Description Cleaned: ")
print(df_clean['Description'].tolist())
print()

 # PHONE NUMBER CLEANING

print("="*70)
print("PHONE NUMBER CLEANING")
print("="*70)

def clean_phone(phone):
    """Standardize phone numbers to 10 digits"""
    # Remove all non-digit characters
    digits_only = re.sub(r'\D', '', str(phone))

    # Handle country code +91
    if len(digits_only) == 12 and digits_only.startswith('91'):
        digits_only = digits_only[2:]
    
    # Validate 10 digits
    if len(digits_only) == 10:
        return digits_only
    else:
        return np.nan
df_clean['Phone_clean'] = df['Phone'].apply(clean_phone)
print("Phone numbers cleaned: ")
print(df_clean[['Phone', 'Phone_clean']])
print()

  # EMAIL VALIDATION

print("="*70)
print("EMAIL VALIDATION")
print("="*70)

def validate_email(email):
    """Check if email format is valid"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    if pd.isna(email):
        return False
    return bool(re.match(pattern, str(email)))

df_clean['Email_valid'] = df_clean['Email'].apply(validate_email)
print("Email validation: ")
print(df_clean[['Email','Email_valid']])
print()

# STRING EXTRACTION

print("="*70)
print("STRING EXTRACTION")
print("="*70)

# Extract first name and Last name
df_clean[['First_Name', 'Last_Name']] = df_clean['Name'].str.split(' ', n=1, expand=True)
print("Name split: ")
print(df_clean[['Name', 'First_Name', 'Last_Name']])
print()

# Extract domain from email
df_clean['Email_domain'] = df_clean['Email'].str.extract(r'@([^@]+$)')
print("Email domains: ")
print(df_clean[['Email', 'Email_domain']])
print()

# STRING CONTAINS / STARTSWITH / ENDSWITH

print("="*70)
print("STRING FILTERING")
print("="*70)

# Engineers only
engineers = df_clean[df_clean['Description'].str.contains('Engineer', case=False, na=False)]
print("Engineers: ")
print(engineers[['Name', 'Description']])
print()

# Gmail users
gmail_users = df_clean[df_clean['Email'].str.endswith('@gmail.com', na=False)]
print("Gmail users: ")
print(gmail_users[['Name', 'Email']])
print()

 # CATEGORICAL ENCODING (Previe of ML Prep)

print("="*70)
print("CATEGORICAL ENCODING (ML PREPARATION)")
print("="*70)

# Label encoding
city_codes = pd.Categorical(df_clean['City'])
df_clean['City_Code'] = city_codes.codes
print("City label encoding: ")
print(df_clean[['City', 'City_Code']])
print()

# One-hot encoding
job_dummies = pd.get_dummies(
    df_clean['Description'],
    prefix='Job'
)
print("One-hot encoded jobs: ")
print(job_dummies)