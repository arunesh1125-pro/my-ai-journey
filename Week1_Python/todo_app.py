import json
from datetime import datetime

# Load tasks from file
def load_tasks():
    try:
        with open("tasks.json", "r") as file:
            return json.load(file)
    except FileNotFoundError:
        return []

# Save tasks to file
def save_tasks(tasks):
    with open("tasks.json", "w") as file:
        json.dump(tasks, file, indent=2)

# Add a new task
def add_task(tasks):
    task_name = input("Enter a task: ")
    task = {
        "id": len(tasks) + 1,
        "name": task_name,
        "completed": False,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M")
    }
    tasks.append(task)
    save_tasks(tasks)
    print(f"✅ Task added: {task_name}")

# View all tasks
def view_tasks(tasks):
    if not tasks:
        print("No task found!")
        return
    print("\n" + "="*50)
    print('YOUR TO-DO LIST')
    print("="*50)
    for task in tasks:
        status = "✓" if task["completed"] else "x"
        print(f"{task['id']}.{[status]} {task['name']}")
        print(f"    Created: {task['created']}")
    print("="*50)

# Mark task as complete
def complete_task(tasks):
    view_tasks(tasks)
    try:
        task_id = int(input("\Enter task ID to mark complete: "))
        for task in tasks:
            if task["id"]==task_id:
                task["completed"] = True
                save_tasks(tasks)
                print(f"✅ Task {task_id} marked as complete!")
                return
        print("Task not found!")
    except ValueError:
        print("Invalid input!")

# Delete a task
def delete_task(tasks):
    view_tasks(tasks)
    try:
        task_id = int(input("\nEnter task ID to delete: "))
        for i, task in enumerate(tasks):
            if task["id"] == task_id:
                tasks.pop(i)
                save_tasks(tasks)
                print(f"🗑️ Task {task_id} deleted!")
                return
    except ValueError:
        print("Invalid input!")

# Main Program
def main():
    tasks = load_tasks()

    while True:
        print("\n"+ "="*50)
        print("TO-DO LIST MANAGER")
        print("\n"+ "="*50)
        print("1. Add Tasks")
        print("2. View Tasks")
        print("3. Mark Task Complete")
        print("4. Delete Task")
        print("5. Exit")

        choice = input("\nEnter choice (1-5): ")

        if choice == "1":
            add_task(tasks)
        elif choice == "2":
            view_tasks(tasks)
        elif choice == "3":
            complete_task(tasks)
        elif choice == "4":
            delete_task(tasks)
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice!")

if __name__ == "__main__":
    main()
