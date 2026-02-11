import numpy as np

#Create sample dataset: 100 students, 5 subjects
np.random.seed(42)
scores = np.random.randint(0, 101, size=(100, 5))

print("Student scores (first 10):\n", scores[:10])
print()

#Task 1: Find students who scored > 90 in All subjects
perfect_students = np.all(scores>90, axis=1) # row
print(f"Students with all scores > 90: {np.sum(perfect_students)}")
print()

#Task 2: Find students who failed (<40) in ANY subject
failed_students = np.any(scores<40, axis=1)
print(f"Students who failed any subjects: {np.sum(failed_students)}")
print()

#Task 3: Replace all scores < 35 with 35 (grace marks)
scores_with_grace = np.where(scores<35, 35, scores)
print("Scores after grace marks (first 10):\n", scores_with_grace[:10])
print()

#Task 4: Find subject-wise pass percentage (>=40)
pass_per_subject = np.mean(scores>=40, axis=0)*100
print("Pass percentage per subject: ")
for i, pass_pct in enumerate(pass_per_subject):
    print(f"  Subject {i+1}:  {pass_pct:.1f}%")
print()

#Task 5: find top 10 students by average score
avg = np.mean(scores, axis=1)
top_10 = np.argsort(avg)[-10:][::-1]  #Sort and reverse
print("Top 10 student indicies: ", top_10)
print("Their average scores: ", avg[top_10])
print()

#Task 6: Distribution of grades (A: 90+, B: 75-89, C: 60-74, D: 40-59, F: <40)
avg_scores = np.mean(scores, axis=1)
grade_A = np.sum(avg_scores >= 90)
grade_B = np.sum((avg_scores >= 75) & (avg_scores <90))
grade_C = np.sum((avg_scores >= 60) & (avg_scores <75))
grade_D = np.sum((avg_scores >= 40) & (avg_scores <60))
grade_F = np.sum(avg_scores < 40)

print("Grade Distributions: ")
print(f"    A (90+):    {grade_A}")
print(f"    B (75-89):    {grade_B}")
print(f"    C (60-74):    {grade_C}")
print(f"    D (40-59):    {grade_D}")
print(f"    F (<40):    {grade_F}")