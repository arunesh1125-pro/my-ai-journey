# Task 1: FizzBuzz (classic programming problem!)
# Print numbers 1-100, but:
# - If divisible by 3, print "Fizz"
# - If divisible by 5, print "Buzz"
# - If divisible by both, print "FizzBuzz"
# - Otherwise, print the number

for i in range (1, 101):
    if i%3==0 and i%5==0:
        print("FizzBuzz")
    elif i%3==0:
        print("Fizz")
    elif i%5==0:
        print("Buzz")
    else:
        print(i)

# Task 2: Find prime numbers between 1-50
primes = []
for num in range(2, 51):
    is_prime=True
    for i in range(2, int(num**0.5)+1):
        if num%i==0:
            is_prime=False
            break
    if is_prime:
        primes.append(num)
print(f"Prime numbers:  {primes}")