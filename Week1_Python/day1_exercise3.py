# Task: Create a student management system
students={
    "student1": {"name":"Sandhiya", "marks":[90, 92, 93]},
    "student2": {"name":"Arunesh", "marks":[90, 88, 89]},
    "student3": {"name":"Sneha", "marks":[92,93,95]}
}

#Calculate and add average marks for each student
for student_id, info in students.items():
    marks = info["marks"]
    average = sum(marks) / len(marks)
    info["average"] = average
    print(f"{info['name']}: Average = {average:.2f}")

#Find student with hightest average
best_student = max(students.values(), key=lambda x:x["average"])
print(f"\nTop student: {best_student['name']}")