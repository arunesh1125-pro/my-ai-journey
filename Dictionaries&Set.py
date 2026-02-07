#Dictionaries(Key-value pair - Important for ML)
student = {
    "name": "Sandhiya",
    "age": 20,
    "city": "Chennai",
    "skills": ["Python", "Front-end+Frameworks", "Flask/Django", "My SQL"]
}

#Accessing Values
print(student["name"])
print(student.get("age"))
print(student.get("country", "India"))

#Adding/modifying
student['email'] = "sandhiya.v1909@gmail.com"
student["age"] = 21

#Dictionary operation
print(student.keys()) #All Keys
print(student.values()) # All values
print(student.items()) #All key-value pairs

#Looping through Dictionary
for key, value in student.items():
    print(f"{key}: {value}")

#Count Frequency of Characters in a String
text = "banana"
freq = {}

for x in text:
    if x in freq:
        freq[x] += 1
    else:
        freq[x] = 1
print(freq)

#Find the Student with higest marks
marks = {
    "Arunesh":85,
    "Sandhiya":93,
    "Sneha":92
}
highest = max(marks, key=marks.get)
print("Topper:", highest)
#Merge two cictionaries
dict1 = {"a":1, "b":2}
dict2 = {"c":1, "d":3}
merged = dict1 | dict2
print(merged)
#Remove Duplicate Values from Dictionary
result={}
seen = set()
for key, value in merged.items():
    if value not in seen:
        result[key]=value
        seen.add(value)
print(seen) #Values from dict
print(result)

#SET (Unique Values only, unordered)
unique_num = {1, 2, 3, 4, 5}
unique_num.add(6)
#unique_num(3) #Won't add duplicate
print(unique_num)

# Set Operation (Useful for data Cleaning!)
set1= {1,2,3,4}
set2={3,4,5,6}
print(set1.union(set2))
print(set1.intersection(set2))
print(set1.difference(set2))