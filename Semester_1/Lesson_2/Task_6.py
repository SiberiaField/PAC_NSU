text = open('input.txt').read().split('\n')
symbols = 0
words = 0
new_word = True
for str in text:
    length = len(str)
    i = 0
    while str[i] == ' ':
        i += 1
    while i < length:
        if str[i] == ' ':
            new_word = True
        else:
            if new_word:
                words += 1
                new_word = False
            symbols += 1
        i += 1
    new_word = True
print(len(text), words, symbols)
        
