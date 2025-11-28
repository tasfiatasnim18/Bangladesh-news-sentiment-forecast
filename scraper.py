#!/usr/bin/env python
# coding: utf-8

# The_Business_Standard_Newspaper_scrape

# In[1]:


import requests
from bs4 import BeautifulSoup

# URL of the page you want
url = 'https://www.tbsnews.net/archive/2025/09/10'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Send request
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print(f"Failed to fetch page, status code: {response.status_code}")
    exit()

# Parse HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Find all headlines on this page
headlines = soup.find_all('h4', class_='card-title')  # adjust if needed

# Extract text
headlines_text = [h.text.strip() for h in headlines]

# Print
print(f"Total headlines found: {len(headlines_text)}\n")
for i, h in enumerate(headlines_text, 1):
    print(f"{i}. {h}")

# Save to file
with open('tbs_headlines_2025_08_01.txt', 'w', encoding='utf-8') as f:
    for h in headlines_text:
        f.write(h + '\n')

print("\nHeadlines saved to tbs_headlines_2025_08_01.txt")


# NEWAGE_Newspaper_Scrape

# In[2]:


import requests
from bs4 import BeautifulSoup

# URL of the page you want
url = 'https://www.newagebd.net/archive?date=2025-08-01'

headers = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
}

# Send request
response = requests.get(url, headers=headers)
if response.status_code != 200:
    print(f"Failed to fetch page, status code: {response.status_code}")
    exit()

# Parse HTML
soup = BeautifulSoup(response.content, 'html.parser')

# Find all headlines on this page
headlines = soup.find_all('h2', class_='card-title h5 h4-sm h3-lg')  # adjust if needed

# Extract text
headlines_text = [h.text.strip() for h in headlines]

# Print
print(f"Total headlines found: {len(headlines_text)}\n")
for i, h in enumerate(headlines_text, 1):
    print(f"{i}. {h}")

# Save to file
with open('newage_headlines_2025_08_01.txt', 'w', encoding='utf-8') as f:
    for h in headlines_text:
        f.write(h + '\n')

print("\nHeadlines saved to newage_headlines_2025_08_01.txt")


# In[ ]:




