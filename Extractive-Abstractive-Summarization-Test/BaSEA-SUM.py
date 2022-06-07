#!/usr/bin/env python
# coding: utf-8

# # 0. Import Modules

# In[1]:


from summarizer import Summarizer
import re
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import pipeline


# # 1. Text-Corpus

# 1. user input asks for the name of the author #this is to strip it from the corpus afterwards
# 2. actual text-file is being loaded
# 3. all of the breaks will be removed from text-corpus #to safe resoruces, and those are not relevant for the summary
# 4. tokenization is avoided at this point, because that will change the integrity of the text

# In[2]:


name = input("Enter the authors name: ")

with open('HP1') as f:
    text = f.read()
text_no_num = re.sub(r'\d+','', text)   
body = text_no_num.replace("\n", " ").replace(name, "")
print(body)


# In[3]:


word_list = body.split()

number_of_words_summarized = len(word_list)

print(number_of_words_summarized)


# # 2. Extractive Summarization

# 1. text is going to be splitted in even chunks / or chapters.
# 2. each chunk is going to be summarized
# 3. summaries will be saved together in a new variable

# In[4]:


batches = body.split("CHAPTER")
    
model = Summarizer()
results = [model(batch, ratio=0.15) for batch in batches]


# In[5]:


extractive = ' '.join(results)
print(extractive)


# In[6]:


model = Summarizer()
result = model(extractive, ratio=0.10)
#extractive_compressed = ''.join(result)
print(result)


# In[7]:


word_list = extractive_compressed.split()

number_of_words_summarized = len(word_list)

print(number_of_words_summarized)


# # 3. Abstractive Summarization

# 1. Abstractive Summarization Model will be executed
# 2. Results should be printed and evaluated

# In[18]:


_model = AutoModelForSeq2SeqLM.from_pretrained(
                "pszemraj/bigbird-pegasus-large-K-booksum",
                low_cpu_mem_usage=True,
            )

_tokenizer = AutoTokenizer.from_pretrained(
                "pszemraj/bigbird-pegasus-large-K-booksum",
            )
                                           

summarizer = pipeline(
                    "summarization", 
                    model=_model, 
                    tokenizer=_tokenizer
                )


# In[19]:


wall_of_text = extractive_compressed

result = summarizer(
           wall_of_text,
           min_length=16, 
           max_length=256,
           no_repeat_ngram_size=3, 
           clean_up_tokenization_spaces=True,
    )


# In[1]:


extr_sum = result['summary_text'])


# In[21]:


word_list = result[0]['summary_text'].split()

number_of_words_summarized = len(word_list)

print(number_of_words_summarized)


# In[ ]:


with open('BaSEA-SUM.txt', 'w', encoding='utf-8') as t:
    t.write(full)

