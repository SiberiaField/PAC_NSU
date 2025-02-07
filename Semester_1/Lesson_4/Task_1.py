import numpy as np
def freqSort(lst: list[int]) -> None:
    unique, freq = np.unique(lst, return_counts=True)
    freq_dict = dict(zip(unique, freq))
    print(freq_dict)
    lst.sort(key= lambda x: -freq_dict[x])

