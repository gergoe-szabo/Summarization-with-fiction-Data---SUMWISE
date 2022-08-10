#!/usr/bin/env python
# coding: utf-8

# GAMEPLAN:
# 
# 1. split input in chapters
# 2. split chapters in batches of 1000 words save batches by chapter
# 3. summarize each batch and save into a file by chapter
# 4. summarize each summarized chapter again and save all chapters into a file
# 5. summarize the resulted file
# 6. use several summarizer models
# 
# BONUS experiment with abstractive and extractive methods for the first summarization step

# In[ ]:


#load modules
import re
import pandas as pd


# In[ ]:


#load input #strip authors name via user input (maybe change later)

name = "J.K. Rowling"
title = "Harry Potter and the Sorcerer's Stone   "
numbers = ["ONE", "TWO", "THREE", "FOUR", "FIVE", "SIX", "SEVEN", "EIGHT", "NINE", "TEN", "ELEVEN", "TWELVE", "THIRTEEN", "FOURTEEN", "FIFTEEN", "SIXTEEN", "SEVENTEEN", "TEEN"]
chapter_titles = ["THE BOY WHO LIVED", "THE VANISHING GLASS", "THE LETTERS FROM NO ONE", "THE KEEPER OF THE KEYS", "DIAGON ALLEY", "THE JOURNEY FROM PLATFORM NINE AND THREE-QUARTERS", "THE SORTING HAT", "THE POTIONS MASTER", "THE MIDNIGHT DUEL", "HALLOWEEN", "QUIDDITCH", "THE MIRROR OF ERISED", "NICOLAS FLAMEL", "NORBERT THE NORWEGIAN RIDGEBACK", "THE FORIBIDDEN FOREST", "THROUGH THE TRAPDOOR", "THE MAN WITH TWO FACES"]

with open('HP1new') as f:
    text = f.read()
pre_text = re.sub(r'\d+','', text)   
body = pre_text.replace("\n", " ").replace(name, "").replace('said', "").replace('told', "").replace(title, "")

#print(body)

#Get rid of chapter titles
for chapter_title in chapter_titles:
    body = body.replace("" + chapter_title + "", "")
    
#Get rid of page numbers
for number in numbers:
    body = body.replace("" + number + "", "")
    
#remove all sentences in qoutation marks    
no_quotes = re.sub(r'".+?"', '', body)

print(no_quotes)


# In[ ]:


word_list = no_quotes.split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


word_list = body.split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


#split in chapters

chapters = no_quotes.split("CHAPTER")
#del chapters[0]

df = pd.DataFrame(chapters)
df


# In[ ]:


word_list = chapters[1].split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


#split chapters in chunks of 500 each
del chapters[0]


chapter_chunks=[]
for text in chapters:
  text = text.split()
  n = 500
  chapter_chunks.append([' '.join(text[i:i+n]) for i in range(0,len(text),n)]) 


# In[ ]:


df = pd.DataFrame(chapter_chunks)
df


# In[ ]:


#print(chapter_chunks[16][0])

print(chapter_chunks[16])


# In[ ]:


word_list = chapter_chunks[1][2].split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


#split book into chunks of 500

book = no_quotes
book = book.split()
n = 500
book_chunks = [' '.join(book[i:i+n]) for i in range(0,len(book),n)]


# In[ ]:


df = pd.DataFrame(book_chunks)
df


# In[ ]:


word_list = book_chunks[78].split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


#load modules for summarization
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline


# In[ ]:


#set parameters and build summarization pipelines

max_source_length = 1024
max_target_length = 512
min_target_length = 124

hf_name = 't5-small'
model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
tokenizer = AutoTokenizer.from_pretrained(hf_name, padding='longest', max_length=max_source_length, truncation=True)

summarizer = pipeline('summarization', model=model, tokenizer=tokenizer, truncation=True, max_length=max_source_length)

#summarize each batch by chapter

chapter_summaries = []
for chapter in chapter_chunks:
    summary = []
    for chunk in chapter:
        summary.append(summarizer(chunk, max_length=max_target_length, min_length=min_target_length, truncation=True)[0]["summary_text"])
    
    chapter_summaries.append(summarizer("\n".join(summary), max_length=max_target_length, min_length=min_target_length, truncation=True)[0]["summary_text"])


# In[ ]:


n = 6

print(chapter_summaries[n])

word_list = chapter_summaries[n].split()

number_of_words = len(word_list)
print(number_of_words)

df = pd.DataFrame(chapter_summaries)
df


# In[ ]:


text = " ".join(map(str, chapter_summaries))
summarized_chapters = text.replace("\n", " ").replace('"', "").replace(" .", ".")
print(summarized_chapters)


# In[ ]:


word_list = summarized_chapters.split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


#set parameters and build summarization pipelines

max_source_length = 1024
max_target_length = 512
min_target_length = 124

hf_name = 't5-small'
model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
tokenizer = AutoTokenizer.from_pretrained(hf_name, padding='longest', max_length=max_source_length, truncation=True)

summarizer = pipeline('summarization', model=model, tokenizer=tokenizer, truncation=True, max_length=max_source_length)


#summarize each chapter

summaries = []
for batch in chapter_chunks:
    summaries.append(summarizer(batch, max_length=max_target_length, min_length=min_target_length, truncation=True))


# In[ ]:


#safe results of chapter summarization into kondensed chapters

results = []

for summary in summaries:
    results.append("\n".join(s["summary_text"] for s in summary))
    
    
df = pd.DataFrame(results)
df


# In[ ]:


text = " ".join(map(str, results))
all_summaries = text.replace("\n", " ").replace('"', "").replace(" .", ".")


# In[ ]:


word_list = results[3].split()

number_of_words = len(word_list)

print(number_of_words)
print(all_summaries)
print(results[3])


# In[ ]:


#set parameters and build summarization pipelines

max_source_length = 500
max_target_length = 100
min_target_length = 80

hf_name = 't5-small'
model = AutoModelForSeq2SeqLM.from_pretrained(hf_name)
tokenizer = AutoTokenizer.from_pretrained(hf_name, padding='longest', max_length=max_source_length, truncation=True)

summarizer = pipeline('summarization', model=model, tokenizer=tokenizer, truncation=True, max_length=max_source_length)


#summarize each resulted chapter

goal = []
for result in results:
    goal.append(summarizer(result, max_length=max_target_length, min_length=min_target_length, truncation=True))


# In[ ]:


print(goal[3])


# In[ ]:


final = []

for element in goal:
    final.append("\n".join(s["summary_text"] for s in element))
    
final


# In[ ]:


pre_final = " ".join(map(str, final))
finalfinal = pre_final.replace("\n", "").replace('"', "").replace(".", ".").replace("...", "").replace("..", "").replace(" .", ".")
print(finalfinal)


# In[ ]:


word_list = finalfinal.split()

number_of_words = len(word_list)

print(number_of_words)


# In[ ]:


with open('/model-name-improved-book-summary.txt', 'w', encoding='utf-8') as t:
    t.write(finalfinal)


# In[ ]:


with open('/model-name-quick-book-summary.txt', 'w', encoding='utf-8') as t:
    t.write(summarized_chapters)


# In[ ]:


with open('/model-name-summarized-chapters.txt', 'w', encoding='utf-8') as t:
    t.write(results)

