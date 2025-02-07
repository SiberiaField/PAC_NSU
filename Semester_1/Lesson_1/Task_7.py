import os

main_dir = input()
for _, _, files in os.walk(os.path.abspath(main_dir)):
    for name in files:
        print(os.path.abspath(name))