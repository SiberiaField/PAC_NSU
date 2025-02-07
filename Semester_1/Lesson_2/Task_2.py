str = input().split(" ")

max_word = str[0]
max_length = len(max_word)
for word in str:
    word_lenght = len(word)
    if word_lenght > max_length:
        max_word = word
        max_length = word_lenght

print("Max word is '", max_word, "'", sep = '')
print("His lenght =", max_length)