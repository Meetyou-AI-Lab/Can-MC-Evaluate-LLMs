from collections import Counter
import pandas as pd
import numpy as np

def consistency(sequence):
    # Count the number of occurrences of each character
    freq = Counter(sequence)
    # Find the character that appears the most and how many times it appears
    max_freq_char, _ = freq.most_common(1)[0]  # Returns the most frequent occurrence of the element and its number of times
    # For other characters, subtract 1 for each occurrence, and then add all the values
    total = sum((count if char == max_freq_char else max(0, count - 1)) for char, count in freq.items())
    return total / len(sequence)

def accuracy(sequence):
    freq = Counter(sequence)
    total_D = sum(count if char == 'D'else 0 for char, count in freq.items())
    return total_D / len(sequence)

data = pd.read_csv('CARE-MI-test.csv',encoding='utf-8') 
con = data['Answer'].tolist()
j = -1
for i in con:
    j = j+1
    print(i)
    data['Consistency'].loc[j] = consistency(i)
    data['Accuracy'].loc[j] = accuracy(i)
data.to_csv('CARE-MI-test.csv',index = False,encoding='utf_8_sig')

# print(consistency("BAACD"))
# print(consistency("ABCDE"))
# print(consistency("AABBC")) 
# print(consistency("AAAAABC"))
# print(consistency("aaabbcd"))
# print(consistency("AAABBBC"))