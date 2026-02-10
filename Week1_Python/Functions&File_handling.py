#Basic function
def greet(name):
    return f"Hello, {name}!"
print(greet("Arunesh"))

#Multiple parameters
def add(a, b):
    return a+b
result = add(10, 5)
print(result)

#Default parameters
def power(base, exponent=2):
    return base ** exponent
print(power(5)) #Default - exponent=2 then base=x
print(power(2, 3)) #Here exponent been changed into 3

#Multiple return values
def min_max(numbers):
    return min(numbers), max(numbers)
minimum, maximum = min_max([1, 5, 3, 9, 2])
print(f"Min: {minimum}, Max: {maximum}")

#lambda functions (anonyms, one line functions)
square = lambda x: x**2
print(square(6))

#Useful with map, filter
numbers = [1,2,3,4,5]
squared = list(map(lambda x: x**2, numbers))
print(squared)
evens = list(map(lambda x: x%2==0, numbers))
print(evens)

#Funtion t print prime numbers in range
def print_primes(i, j):
    for num in range(i, j+1):
        if num>1:
            is_prime = True
            for i in range(2, num):
                if num%i == 0:
                    is_prime=False
                    break
            if is_prime:
                print(num, end=" ")
print_primes(1, 20)

#Print Start PAttern
def star_pattern(n):
    for i in range(1, n+1):
        for j in range(i):
            print("*", end=" ")
        print()
star_pattern(4)