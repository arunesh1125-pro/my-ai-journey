#Task : Create a simple contact management system
def add_contact(name, phone):
    with open("contacts.txt", "a") as file:
        file.write(f"{name}, {phone}\n")
    print(f"Contact: {name} added")

def view_contacts():
    try:
        with open("contacts.txt", "r") as file:
            print("\n=== Contacts ===")
            for line in file:
                name, phone = line.strip().split(",")
                print(f"Name: {name}, Phone:{phone}")
    except FileNotFoundError:
        print("No contacts found!")

def search_contact(name):
    try:
        with open("contacts.txt","r") as file:
            for line in file:
                contact_name, phone = line.strip().split(",")
                if contact_name.lower() == name.lower():
                    print(f"Found: {contact_name} - {phone}")
                    return
            print(f"Contact {name} not found")
    except FileNotFoundError:
        print("No contacts found!")

# Menu System
while True:
    print("\n Contact Menu ")
    print("\n1. Add Contacts")
    print("2. View Contacts")
    print("3. Search Contact")
    print("4. Exit")

    choice = input("Enter choice: ")

    if choice == "1":
        name = input('Enter name: ')
        phone = int(input('Enter phone: '))
        add_contact(name, phone)
    elif choice == "2":
        view_contacts()
    elif choice =="3":
        name=input("Enter name to search: ")
        search_contact(name)
    elif choice =="4":
        print("Goodbye !")
        break
    else:
        print("Invalid Choice !")