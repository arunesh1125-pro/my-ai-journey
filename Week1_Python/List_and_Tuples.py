"""#List (Mutable - can be changed)
num = [1, 2, 3, 4, 5]
names = ["Sandhiya", "Arunesh", "Nimishika"]
mixed = [1, "Hi", 3.14, True]

#Accessing elements
print(num[0]) #first element
print(num[-1]) #last element
print(num[1:3]) #slicing

#LO - List operations
num.append(6) #Add to end
num.insert(0, 0) # Insert at pos, val
num.remove(3) #remove value
popped = num.pop() #remove and return last element
print(len(num)) #Length
print(sum(num)) # sum of all elements
print(max(num)) #Maximum value
print(num) """

"""
#LO - Sum of All Elements in a List
numbers = [10, 20, 30, 40]
total = 0

for num in numbers:
    total += num
print("Sum: ", total)

#Count Even and Odd Numbers in a List 
num1 = [1, 2, 3, 4, 5, 6]
even = 0
odd = 0

for num in num1:
    if num % 2 == 0:
        even += 1
    else:
        odd += 1

print("Even:", even)
print("Odd",odd)

#Reverse a List without using reverse()
num2 = [1,2,3,4]
rev_list=[]
for num in num2:
    rev_list = [num] + rev_list

print(rev_list)

#Find the Second Largest Number
num3 = [10, 5, 20, 8]
num3.sort()

print("The Second Largest:", num3[-2])

#Remove All Occurence of an Element
num4 = [1, 2, 3, 2, 4, 2]
new_li = []

for num in num4:
    if num != 2:
        new_li.append(num)
print(new_li)
"""
"""#LC - list comprehension (Important for Ml)
squares = [x**2 for x in range(10)]
print(squares)
evens = [x for x in range(20) if x%2 == 0]
print(evens)

#Convert String to Uppercase
names = ["Python", "Java", "c"]
upper_names = [name.upper() for name in names]
print(upper_names)

#Get Length of Each Word in a List
wrds = ["apple", "banana", "cherry"]
lengths = [len(word) for word in wrds]
print(lengths)

#Filter Number Greater than 5
num1 = [2, 4, 6, 8, 1, 9]
res = [num for num in num1 if num > 5]
print(res)

#Square Only Even Numbers
num2 = [1,2,3,4,5,6]
res1 = [num*num for num in num2 if num%2==0]
print(res1)

#Convert Celsius to Fahrenheit
celsius = [0, 10, 20, 30]
fahrenheit = [(temp *9/5)+32 for temp in celsius]
print(fahrenheit)

#Extraxt Numbers Greater than Average
num3 = [10,20,30,40]
avg = sum(num3)/len(num3)
res2 = [x for x in num3 if x > avg]
print(res2)

#Get First Letter of Eaxh Word
words = ["Python", "Java", "C"]
letters = [x[0] for x in words]
print(letters)

#Remove Vowels from Words
wds = ["apple", "banana", "orange"]
vowels = "aeiou"

res3 = ["".join(ch for ch in word if ch not in vowels)for word in wds]
print(res3) """

#Tuples (immutable - cannot be changed)
coordinates = (10.5, 20.3)
rgb = (255, 128, 0)

for num in rgb:
    print(num)
print(len(rgb))
#Accessing (same as lists)
print(coordinates[0])
print(coordinates[0:2]) #Slicing can be performed

#Check if elemt exists in tuple
num1 = (1,2,2,3,4,5)
if 3 in num1:
    print("Element found")
else:
    print("Element not found")
print("Count of 2: ", num1.count(2))
print("Index of 2: ",num1.index(2))
print(max(num1))
print(min(num1))

#Convert tuple into list ad Modify it
colors = ("red", "blue", "green")
temp_list = list(colors)
temp_list.append("yellow")

colors = tuple(temp_list)
print(colors)

#Unpack Elements of Tuple
student = ("Arunesh", 21, "AI")
name, age, course = student #Like a swapping temp var
print(name)
print(age)
print(course)

#Find the Sum of Tuple Elements using a Loop
num2=(1,2,3,4,5)
sum=0
for num in num2:
    sum += num
print("Sum",sum)

#Count even and Odd Numbers in Tuple
num3 = (10,15,20,25,30)
even = 0
odd = 0
for num in num3:
    if num % 2 == 0:
        even += 1
    else:
        odd += 1
print("Even", even)
print("Odd", odd)