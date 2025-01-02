from nltk import word_tokenize, pos_tag

text = "It's full of hearts, and they're blowing up."
words = word_tokenize(text)
print(f"Words: {words}")

tagged = pos_tag(words)
print(f"POS tagging: {tagged}")