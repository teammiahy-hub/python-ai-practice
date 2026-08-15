print("Hello world")
print("I am learning Python")
print("I will become an AI engineer")

name = "Faniry"
age = 33
goal = "engineer"
print(f"My name is {name}")
print(f"I am {age}")
print(f"My goal is to become an {goal}")

skills = {"RAG","3gpp","langchain"}
for skill in skills:
    print(f"{skill}")

print(f"len of skills {len(skills)}")

def greet_persons(name,role):
    message = f"Hello {name}, I am an {role}"
    return message

message = greet_persons("Faniry","Engineers")
print(message)