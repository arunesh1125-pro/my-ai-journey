#IF-ELSE STATEMENTS
#1
age = 21
if age >= 18:
    print("You can vote in Election")
elif age>= 16:
    print("You can get a LLR License")
else:
    print("You're too young")
#2 Num is even or odd
num = 7
if num%2 == 0:
    print("Even")
else:
    print("Odd")
#3 FInd Greatest num
a=10
b=20
if a>b:
    print("a is greatest")
else:
    print("b is greatest")
#Gretest of 3 num
c=15
if a>=b and a>=c:
    print("a is greatest")
elif b>=a and b>=c:
    print("b is greatest")
else:
    print("c is greatest")
#5 Check Leap Year
year = 2026
if (year%400==0)or(year%4==0 and year%100 !=0):
    print("Leap year")
else:
    print("Not a Leap Year")
#Electricity Bill Calculation
units = 250
bill = 0
if units <= 100:
    bill = units*1
elif units <=200:
    bill = (100*1) + (units-100)*2
else:
    bill = (100*1) + (100*2)+ (units-200)*3
print('Electricity Bill: ₹', bill)

#Simple Login System
username = "admin"
password = "1234"

if username == "admin":
    if password == "1234":
        print("Login Successfully")
    else:
        print("Incorrect password")
else:
    print("Invalid Username")

#For loops
for i in range(5):
    print(i)
for i in range(2, 10, 2): #Start, Stop, Step
    print(i)
fruits = ["apple","banana","orange"]
for fruit in fruits:
    print(fruit)

#Enumerate (get index + value)
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")

#find the Sum of Gigits in Number
num1 = 2345
total = 0
for digit in str(num1):
    total += int(digit)
print("Sum of digits:", total)
#Count prime numbers
count=0
for num in range(2, 21):
    is_prime = True
    for i in range(2,num):
        if num%i == 0:
            is_prime = False
            break
    if is_prime:
        count += 1
print("Number of primes:", count)

#Print a Number Pattern
for i in range(1, 5):
    for j in range(1, i+1):
        print(j, end="")
    print()
#WHILE loops
count = 0
while count<5:
     print(count)
     count+=1
#Reverse a Number Using While Loop
num2=1234
rev = 0
while num2>0:
    digit1 = num2%10
    rev = rev*10 + digit1
    num2= num2//10
print("Reversed number: ", rev)

#Check Number is a Palindrome
num3=121
org = num3
rev1=0
while num3>0:
    dig = num3%10
    rev1=rev1*10 + dig
    num3//=10
if org==rev1:
    print("Palindrome")
else:
    print("Not a Palindrome")
#Find a Factorial
a = 5
fact = 1
while a>0:
    fact *= a
    a -= 1
print("Factorial:", fact)

#Break and continue
for i in range(10):
    if i == 5:
        break #Exit the loop
    if i%2==0:
        continue #Skip to next iteration
    print(i)