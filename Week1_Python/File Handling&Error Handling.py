"""#Writing to a file
with open("data.txt", "w") as file:
    file.write("Hello, World!\n")
    file.write("This is line 2\n")

#READING from a file
with open("data.txt", "r") as file:
    content = file.read()
    print(content)

#Reading line by line
with open("data.txt","r") as file:
    for line in file:
        print(line.strip())

#Appending to a file
with open("data.txt", "a") as file:
    file.write("This is a new line\n")

# Working with CSV files (important for ML!)
# Writing CSV
data = [
    ["Name", "Age", "City"],
    ["Ram", "56", "Kallakurichi"],
    ["Sneha", "24", "Namakkal"]
]

with open("students.csv", "w") as file:
    for row in data:
        file.write(",".join(row) + "\n")

#Reading CSV
with open("students.csv","r") as file:
    for line in file:
        values = line.strip().split(",")
        print(values)

#Writing multiple lines into a file
file = open("datas.txt","w")
lines = ["Python\n", "File Handling\n", "For Learning\n"]
file.writelines(lines)
file.close()
print("Data written successfully")

#Read file and count no. of files
file=open("datas.txt", "r")
count=0
for line in file:
    count+=1
file.close()
print("Number of lines: ", count)

#Copy content from one file to Another
source = open("data.txt", "r")
destination = open("datas.txt", "w")

for line in source:
    destination.write(line)
source.close()
destination.close()
print("File copied successfully")

#Count Words in a File
file = open("data.txt","r")
word_count=0
for line in file:
    words = line.split()
    word_count += len(words)
file.close()
print("Total words:", word_count) """

#ERROR HANDLING (try-except)
try:
    number = int(input("Enter a number: "))
    result = 10/number
    print(f"Result: {result}")
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("Cannot divide by zero!")
except Exception as e:
    print(f"An error occured : {e}")
finally:
    print("This always executes")

#File Handling with Error Handling
try:
    file=open("data.txt","r")
    print(file.read())
except FileNotFoundError:
    print("Error: File Not Found")
except IOError:
    print("Error: File cannot be read")
finally:
    try:
        file.close()
    except:
        pass

#Password Validation with Error Handling
try:
    password = input("Enter password: ")

    if len(password) < 6:
        raise ValueError("Password too short, Enter 8 Digit Password")
    print("Password accepted")
except ValueError as e:
    print("Error", e)
else:
    print("Validation successful")