"""
CAPSTONE PROJECT: Professional Data Cleaning Pipeline
Clean a completely messy dataset from raw to ML-ready!
This demonstrates your full data cleaning skills.
"""
import pandas as pd
import numpy as np
import re
from datetime import datetime

print("="*70)
print("PROFESSIONAL DATA CLEANING PIPELINE")
print("="*70)

 # STEP 0: CREATE MESSY DATASET

print("\n📥 STEP 0: Loading Raw Data...")

np.random.seed(42)

# Simulate extremely messy real-world HR dataset
raw_data = {
    'emp_id': ['E001', 'E002', 'E003', 'E002', 'E004', 'E005',
               'E006', None, 'E008', 'E009', 'E010'],
    'employee_name': ['  Alice Johnson  ', 'BOB Smith', 'charlie brown',
                      'BOB Smith', 'DAVID LEE', 'eve Wilson ',
                      'Frank O Brien', 'Grace kim', '  Henry  Park  ',
                      None, 'Jack Chen'],
    'age': ['25', '30', '35', '30', '-5', '28',
            '150', '32', '29', '31', 'twenty-seven'],
    'department': ['IT', 'hr', 'IT', 'hr', 'Finance',
                   'HR', 'it', 'Finance', None, 'IT', 'FINANCE'],
    'salary': ['50,000', '60000', '75,000', '60000', '55000',
               '70,000', '65000', '-1000', '80,000', '72000', None],
    'join_date': ['2021-01-15', '15/02/2021', 'March 3, 2021',
                  '15/02/2021', '04-04-2021', '2021-05-10',
                  '06/2021/15', 'July 7 2021', '2021-08-20',
                  '2021-09-01', '2021-10-15'],
    'email': ['alice.j@company.com', 'BOB.smith@Company.COM',
              'charlie.b@company.com', 'BOB.smith@Company.COM',
              'not_an_email', 'eve.w@company.com',
              'frank.ob@company.com', 'grace.k@company.com',
              'henry.p@company.com', None, 'jack.c@company.com'],
    'phone': ['+91 98765 43210', '9876543211', '(987) 654-3212',
              '9876543211', '98765-43213', '91-9876543214',
              'not-a-phone', '9876543216', '9876543217',
              '9876543218', '9876543219'],
    'performance_score': [4.5, 3.8, 4.2, 3.8, None, 4.7,
                          3.5, 4.0, None, 3.9, 4.1],
    'city': ['Chennai', 'MUMBAI', ' Delhi ', 'MUMBAI', 'bangalore',
             'Kolkata', 'HYDERABAD', 'Pune ', ' Chennai ', 'Mumbai', 'Delhi']
}


