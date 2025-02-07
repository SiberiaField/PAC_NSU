word = input()

lenght = len(word)
is_palindrome = True
for i in range(0, (int)(lenght / 2)):
    j = lenght - 1 - i
    if word[i] != word[j]:
        is_palindrome = False
        break

if is_palindrome:
    print("Yes")
else:
    print("No")
