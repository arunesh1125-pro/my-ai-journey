 # Task 1: Create a function to calculate factorial
def factorial(n):
    if n==0 or n==1:
        return 1
    result = 1
    for i in range(2, n+1):
        result *= i
    return result
print(factorial(5))

# Task 2: Create a funtion to check if a string is palindrome
def is_palindrome(text):
    text = text.lower().replace(" ", "")
    return text == text[::-1]
print(is_palindrome("hello"))
print(is_palindrome("racecar"))
print(is_palindrome("Malayalam"))
print(is_palindrome("A man a plan a canal Panama"))

#Task3: Create a function to find nth Fibonacci number
"""def fibonacci(n):
    if n<=1:
        return n
    a, b = 0, 1
    for _ in range(2, n+1):
        a, b = b + a
    return b
print(fibonacci(10))"""

def fibonacci(n):
    if n <= 1:
        return n
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    return b

print(fibonacci(10))  # 5