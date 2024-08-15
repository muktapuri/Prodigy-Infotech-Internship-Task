#!/usr/bin/env python
# coding: utf-8

# In[11]:


#prodigy_task/ab.txt
def preprocess_text(text):
    text = text.lower()  # Convert to lower case
    text = text.replace('\n', ' ')  # Replace newlines with spaces
    text = text.strip()  # Remove leading and trailing spaces
    return text

with open(r"C:\Users\Mukta\OneDrive\Desktop\prodigy_internship_task\ab.txt") as file:
    text = preprocess_text(file.read())


# In[13]:


#Build the Markov Chain Model
from collections import defaultdict, Counter

def build_markov_chain(text, n=1):
    markov_chain = defaultdict(Counter)
    
    for i in range(len(text) - n):
        context = text[i:i+n]
        next_char = text[i+n]
        markov_chain[context][next_char] += 1
    
    return markov_chain


# In[15]:


#Calculate Transition Probabilities:
def calculate_probabilities(markov_chain):
    probabilities = {}
    
    for context, counter in markov_chain.items():
        total_count = sum(counter.values())
        probabilities[context] = {char: count / total_count for char, count in counter.items()}
    
    return probabilities


# In[17]:


#Generate Text
#Define the Text Generation Function:
import random

def generate_text(probabilities, start_context, length=100):
    current_context = start_context
    generated_text = current_context

    for _ in range(length):
        next_chars = probabilities.get(current_context, {})
        if not next_chars:
            break
        next_char = random.choices(list(next_chars.keys()), weights=next_chars.values())[0]
        generated_text += next_char
        current_context = generated_text[-len(start_context):]

    return generated_text


# In[19]:


#Use the Model
n = 1  # Using a 1st-order Markov chain
markov_chain = build_markov_chain(text, n)
probabilities = calculate_probabilities(markov_chain)

start_context = text[:n]  # Starting context for generation
generated_text = generate_text(probabilities, start_context, length=200)

print(generated_text)


# In[ ]:




