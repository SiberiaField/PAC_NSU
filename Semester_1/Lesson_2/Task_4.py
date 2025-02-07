str = input().split()
synonyms = {'ggg' : 'hh', 'tt' : 'y'}

length = len(str)
for i in range(0, length):
    syn = synonyms.get(str[i], 0)
    if syn:
        str[i] = syn
print(*str)