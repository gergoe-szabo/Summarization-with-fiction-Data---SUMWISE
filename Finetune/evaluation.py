#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from transformers import pipeline


# In[ ]:


summarize = pipeline("summarization", model="pfad zum modell", tokenizer="pfad zum modell")


# In[ ]:


with open("/home/g-ball/Desktop/Projects/THESIS/Booksum/Dataset/booksum_test.json", "r", encoding="utf-8") as f:
    test_data = json.load(f)


# In[ ]:


predictions = []
summaries = []


for text, summary in zip(test_data["text"], test_data["summary"]):
    predictions.append(summarize(text)[0]["summary_text"])
    summaries.append(summary)


# In[ ]:


summaries


# In[ ]:


predictions


# In[ ]:


from datasets import load_metric

metric = load_metric("rouge")

def calc_rouge_scores(candidates, references):
    result = metric.compute(predictions=candidates, references=references, use_stemmer=True)
    result = {key: round(value.mid.fmeasure * 100, 1) for key, value in result.items()}
    return result


# In[ ]:


calc_rouge_scores(predictions, summaries)

with open('evaluation-results.txt', 'w', encoding='utf-8') as t:
    t.write(calc_rouge_scores(predictions, summaries))

