#!/usr/bin/env python
# coding: utf-8

# Import Dependencies

# In[55]:


import string
import nltk
from nltk.corpus import stopwords, opinion_lexicon


# Download required NLTK data (only once)

# In[86]:


#nltk.download("stopwords")
#nltk.download('opinion_lexicon')


# Stopwords (includes prepositions)

# In[57]:


stop_words = set(stopwords.words("english"))


# Sentiment word lists

# In[58]:


positive_words = set(opinion_lexicon.positive())
negative_words = set(opinion_lexicon.negative())


# File Path 

# In[87]:


file_path = "data.txt"


# Load headlines from a text file

# In[88]:


def load_headlines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        headlines = f.readlines()
    # remove blank lines
    headlines = [h.strip() for h in headlines if h.strip()]
    return headlines


# Clean text and tokenize

# In[89]:


def clean_and_tokenize(text):
    # remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # lowercase
    text = text.lower()
    # tokenize
    tokens = text.split()
    # remove stopwords (includes prepositions)
    tokens = [w for w in tokens if w not in stop_words]
    return tokens


# Classify words into Positive, Negative, Neutral

# In[90]:


def classify_words(tokens):
    pos, neg, neu = [], [], []
    for word in tokens:
        if word in positive_words:
            pos.append(word)
        elif word in negative_words:
            neg.append(word)
        else:
            neu.append(word)
    return pos, neg, neu


# Main processing

# In[102]:


# Process headlines and summarize
def summarize_and_predict(file_path):
    headlines = load_headlines(file_path)

    total_words = 0
    total_pos = 0
    total_neg = 0
    total_neu = 0

    for h in headlines:
        tokens = clean_and_tokenize(h)
        pos, neg, neu = classify_words(tokens)
        total_words += len(tokens)
        total_pos += len(pos)
        total_neg += len(neg)
        total_neu += len(neu)

    # Print summary
    print("=== SUMMARY ===")
    print("Total headlines   :", len(headlines))
    print("Total words       :", total_words)
    print("Positive words    :", total_pos)
    print("Negative words    :", total_neg)
    print("Neutral words     :", total_neu)

summarize_and_predict(file_path)


# Counting Topicwise Sentiment words

# In[92]:


import re
from collections import defaultdict

# --- Define sentiment lexicon ---
positive_words = {"good", "growth", "peace", "happy", "progress", "success"}
negative_words = {"bad", "crime", "corruption", "violence", "fail", "crisis"}

#clean text & tokenize ---
def clean_tokens(text):
    text = text.lower()
    return re.findall(r'\b[a-z]+\b', text)

#classify tokens into pos/neg ---
def classify(tokens):
    pos = sum(1 for t in tokens if t in positive_words)
    neg = sum(1 for t in tokens if t in negative_words)
    return pos, neg

#topic words finder ---
def topic_words(headlines, seed, top_n=50):
    co_occur = defaultdict(int)
    for h in headlines:
        tokens = clean_tokens(h)
        if seed in tokens:
            for w in tokens:
                if w != seed:
                    co_occur[w] += 1
    return [w for w, _ in sorted(co_occur.items(), key=lambda x: -x[1])[:top_n]]

#filter headlines by topic words ---
def filter_headlines(headlines, words):
    return [h for h in headlines if any(w in clean_tokens(h) for w in words)]

#topic sentiment counts ---
def topic_sentiment_counts(headlines, seed):
    words = topic_words(headlines, seed)
    filtered = filter_headlines(headlines, words)

    pos_count, neg_count, neu_count = 0, 0, 0
    for h in filtered:
        tokens = clean_tokens(h)
        pos, neg = classify(tokens)
        neu = len(tokens) - (pos + neg)
        pos_count += pos
        neg_count += neg
        neu_count += neu

    return pos_count, neg_count, neu_count

# --- Load headlines ---
def load_headlines(file_path):
    with open(file_path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

file_path = "data.txt"
headlines = load_headlines(file_path)

# --- Run topic sentiment analysis ---
topics = {"Economic": "economy", "Crime": "crime", "Political": "politics"}

for name, seed in topics.items():
    pos, neg, neu = topic_sentiment_counts(headlines, seed)
    print(f"{name} Topic Sentiment:")
    print(f"  Positive words: {pos}")
    print(f"  Negative words: {neg}")
    print(f"  Neutral words:  {neu}\n")


# Predict country condition

# In[93]:


def predict_condition(pos, neg, total_words=None):
    if total_words:  # include neutral words
        score = (pos - neg) / total_words
    else:  # sentiment-only
        total_sentiment = pos + neg
        if total_sentiment == 0: return "No data"
        score = (pos - neg) / total_sentiment

    if score > 0.05: return "Positive condition"
    elif score < -0.05: return "Negative condition"
    else: return "Neutral condition"

def topic_words(headlines, seed, top_n=50):
    co_occur = defaultdict(int)
    for h in headlines:
        tokens = clean_tokens(h)
        if seed in tokens:
            for w in tokens:
                if w != seed: co_occur[w] += 1
    return set([w for w, _ in sorted(co_occur.items(), key=lambda x:x[1], reverse=True)[:top_n]])

def filter_headlines(headlines, words):
    return [h for h in headlines if any(w in clean_tokens(h) for w in words)]

# --- Main processing ---
def predict_all(file_path):
    headlines = load_headlines(file_path)

    total_pos, total_neg, total_words = 0, 0, 0
    for h in headlines:
        tokens = clean_tokens(h)
        total_words += len(tokens)
        pos, neg = classify(tokens)
        total_pos += pos
        total_neg += neg

    # Overall country condition
    print("=== Overall Country Condition ===")
    print("Including neutral words:", predict_condition(total_pos, total_neg, total_words))
    print("Sentiment-only words   :", predict_condition(total_pos, total_neg))

    # Topic-specific conditions
    topics = {"Economic":"economy", "Crime":"crime", "Political":"politics"}
    for name, seed in topics.items():
        words = topic_words(headlines, seed)
        filtered = filter_headlines(headlines, words)
        pos, neg = 0, 0
        for h in filtered:
            p, n = classify(clean_tokens(h))
            pos += p
            neg += n
        print(f"Predicted {name} condition:", predict_condition(pos, neg))

# --- Run ---
file_path = "data.txt"  # replace with your file path
predict_all(file_path)


# Predicting Conditions

# In[94]:


#!pip install wordcloud matplotlib


# In[95]:


import matplotlib.pyplot as plt
from wordcloud import WordCloud


# In[96]:


# --- Load headlines ---
headlines = load_headlines("data.txt")


# In[97]:


# --- Functions for word clouds ---
def generate_wordcloud(headlines, title="Word Cloud"):
    text = " ".join(headlines)
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap='tab20', max_words=200).generate(text)
    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(title, fontsize=16)
    plt.show()


# In[98]:


# --- 1️⃣ Overall WordCloud ---
generate_wordcloud(headlines, title="Overall Headlines Word Cloud")


# In[99]:


# --- 2️⃣ Topic WordClouds ---
topics = {"Economic":"economy"}
for name, seed in topics.items():
    words = topic_words(headlines, seed)
    filtered_headlines = filter_headlines(headlines, words)
    generate_wordcloud(filtered_headlines, title=f"{name} Headlines Word Cloud")


# In[100]:


# --- 2️⃣ Topic WordClouds ---
topics = {"Crime":"crime"}
for name, seed in topics.items():
    words = topic_words(headlines, seed)
    filtered_headlines = filter_headlines(headlines, words)
    generate_wordcloud(filtered_headlines, title=f"{name} Headlines Word Cloud")


# In[101]:


# --- 2️⃣ Topic WordClouds ---
topics = {"Political":"politics"}
for name, seed in topics.items():
    words = topic_words(headlines, seed)
    filtered_headlines = filter_headlines(headlines, words)
    generate_wordcloud(filtered_headlines, title=f"{name} Headlines Word Cloud")


# Wordcloud for overall positive word

# In[110]:


all_tokens = []
for h in headlines:
    tokens = clean_tokens(h)
    all_tokens.extend(tokens)


# In[113]:


positive_tokens = [w for w in all_tokens if w in positive_words]

# --- Generate word cloud ---
text = " ".join(positive_tokens)
wc = WordCloud(width=800, height=400, background_color='white',
               colormap='Greens', max_words=200).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Overall Positive Words WordCloud", fontsize=16)
plt.show()


# Wordcloud for overall Negative word

# In[122]:


negative_tokens = [w for w in all_tokens if w in negative_words]

# --- Generate word cloud ---
text = " ".join(negative_tokens)
wc = WordCloud(width=800, height=400, background_color='white',
               colormap='Reds', max_words=200).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Overall Negative Words WordCloud", fontsize=16)
plt.show()


# Wordcloud for overall Neutral word

# In[126]:


neutral_tokens = [w for w in all_tokens if w not in positive_words and w not in negative_words]

# --- Generate WordCloud ---
text = " ".join(neutral_tokens)
wc = WordCloud(width=800, height=400, background_color='white',
               colormap='Greys', max_words=200).generate(text)

plt.figure(figsize=(10,5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title("Overall Neutral Words WordCloud", fontsize=16)
plt.show()


# In[124]:


def topic_sentiment_wordcloud(headlines, topic_seed, sentiment="positive", max_words=200):

    words = topic_words(headlines, topic_seed)
    filtered_headlines = filter_headlines(headlines, words)

    all_tokens = []
    for h in filtered_headlines:
        tokens = clean_tokens(h)
        all_tokens.extend(tokens)

    if sentiment == "positive":
        tokens = [w for w in all_tokens if w in positive_words]
        colormap = "Greens"
        title_sentiment = "Positive"
    elif sentiment == "negative":
        tokens = [w for w in all_tokens if w in negative_words]
        colormap = "Reds"
        title_sentiment = "Negative"
    else:  # Neutral
        tokens = [w for w in all_tokens if w not in positive_words and w not in negative_words]
        colormap = "Greys"
        title_sentiment = "Neutral"


    text = " ".join(tokens)
    wc = WordCloud(width=800, height=400, background_color='white',
                   colormap=colormap, max_words=max_words).generate(text)

    plt.figure(figsize=(10,5))
    plt.imshow(wc, interpolation='bilinear')
    plt.axis('off')
    plt.title(f"{topic_seed.capitalize()} {title_sentiment} Words WordCloud", fontsize=16)
    plt.show()


topics = ["economy"]
for topic in topics:
    topic_sentiment_wordcloud(headlines, topic, sentiment="positive")
    topic_sentiment_wordcloud(headlines, topic, sentiment="negative")
    topic_sentiment_wordcloud(headlines, topic, sentiment="neutral")


# In[123]:


topics = ["crime"]
for topic in topics:
    topic_sentiment_wordcloud(headlines, topic, sentiment="positive")
    topic_sentiment_wordcloud(headlines, topic, sentiment="negative")
    topic_sentiment_wordcloud(headlines, topic, sentiment="neutral")


# In[125]:


topics = ["politics"]
for topic in topics:
    topic_sentiment_wordcloud(headlines, topic, sentiment="positive")
    topic_sentiment_wordcloud(headlines, topic, sentiment="negative")
    topic_sentiment_wordcloud(headlines, topic, sentiment="neutral")


# In[ ]:




