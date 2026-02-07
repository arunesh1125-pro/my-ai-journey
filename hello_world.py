"""print("Hello, AI Journey!")
print("Today is Day 1")

#Variables
name = "Sandhiya" #String
age = 20 #integer
height = 5.3 #float
is_working = True #boolean

#Basic Operations
x = 10
y = 5
print(x+y) #addition
print(x-y) #Subtraction
print(x*y) #Multiplication
print(x/y) #Division
print(x//y) #Floor Division
print(x%y) #Remainder
print(x**y) #Power

#Strings
f_name = "Machine"
l_name = "Learning"
full_name = f_name + " " + l_name
print(full_name.upper()) #MACHINE LEARNING
print(full_name.lower()) #machine learning
print(len(full_name))
rev_text = full_name[::-1]
print(rev_text)
word = "malayalam" #Palindrome String
if word == word[::-1]:
    print("Palindrome")
else:
    print("Not a Palindrome")

text = "python programming" #Count how many vowels in given string
vowels = "aeiou"
count = 0
for ch in text:
     if ch in vowels:
        count += 1

print("No. of vowels:", count)"""

sent = "Python is easy to learn" #Count Words in String
wrd = sent.split()
print(len(sent)) #letters
print("Number of word:", len(wrd)) #words

t1 = "Python is fun"
n_t1 = t1.replace(" ", "-") #Replace a Character in a String
print(n_t1)

#Find Frquency of a Charcter
"""t2 = "apple"
char = "p"
count = 0
for ch in t2:
    if ch == char:
        count += 1
print("Frequency: ", count)"""

#Remove Duplicate Charcters from string
t3 = "programming"
res = ""

for ch in t3:
    if ch not in res:
        res += ch
print(res)
"""Iteration 	ch	ch not in result?	result after operation
1	p	True	p
2	r	True	pr
3	o	True	pro
4	g	True	prog
5	r	False	prog (no change)
6	a	True	proga
7	m	True	progam
8	m	False	progam (no change)
9	i	True	progami
10	n	True	progamin
11	g	False	progamin (no change)"""

#Count Uppercase, Lowercase, Digits & Special Charcters
t4 = "Python3@2026"
upper = lower = digit = special = 0

for ch in t4:
    if ch.isupper():
        upper += 1
    elif ch.islower():
        lower += 1
    elif ch.isdigit():
        digit += 1
    else:
        special += 1
print("Uppercase: ", upper)
print("Lowercase: ", lower)
print("Digits: ", digit)
print("Special Characters: ", special)

#Check if Two Strings Are Anagrams: Two strings are anagrams if they contain the same characters in a different order. Example: listen → silent
s1 = "listen"
s2 = "silent"
#sorted() arranges characters alphabetically. If both sorted strings match → anagram.
if sorted(s1) == sorted(s2):
    print("Anagram")
else:
    print("Not an Anagram")

#Find the Longest Word in Sentence
sen = "Python Programming is very interesting"
wds = sen.split()

long = wds[0]

for word in wds:
    if len(word) > len(long):
        long = word
print("Longest word: ",long)

#Count Frequency of Each Character
t5 = "Happy newyear"
for ch in t5:
    print(ch, ":", t5.count(ch))

#Check if a String is pangram. 
#A pangram contains all letters from a–z at least once.
t6 = "the quick brown fox jumps over the lazy dog"
t6 = t6.lower()

alpha = "abcdefghijklmnopqrstuvwxyz"
is_pangram = True
for ch in alpha:
    if ch not in t6:
        is_pangram = False
        break

if is_pangram:
    print("Pangram")
else:
    print("Not a Pangram")