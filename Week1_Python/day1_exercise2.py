# Task 1: Create a list of your favorite 5 movies
movies = ["Jananayagam", "Parashakthi", "Manaadu", "Coolie", "T3"]

#Task 2: Add two movies
movies.append("Ambikkapathhy")
movies.append("Retro")

#Task 3:Print 3rd movie 
print(f"3rd movie: {movies[2]}")

#task 4: create a list of numbers 1-100 using list comprehension
numbers = [x for x in range(1, 101)]
print(numbers)

#task 5: Filter oly numbers divisbile by 5
div_by_5 = [x for x in numbers if x%5 == 0]
print(div_by_5)