# Bangladesh-news-sentiment-forecasting-model
Analyze monthly Bangladeshi news headlines to predict overall, economic, crime, and political sentiment. Includes web scraping, NLP preprocessing, topic-based sentiment analysis, prediction, and word cloud visualizations to track national trends.
This project analyzes monthly news headlines from two major Bangladeshi newspapers and predicts the overall national sentiment, as well as topic-specific sentiment for Economy, Crime, and Politics.

It combines web scraping, NLP preprocessing, sentiment analysis, and topic-level insights, producing both numerical summaries and visualizations like word clouds.

# 🔍 Project Overview
The project workflow:
Web Scraping → Preprocessing → Sentiment Analysis → Topic Analysis → Monthly Condition Prediction → Visualization

Key Goals:
Understand the general socio-economic mood of the country using news headlines.
Identify trends in economy, crime, and political topics.
Visualize headline trends for quick insights.

# 🛠 Tech Stack

Python 3.x
Libraries: requests, BeautifulSoup4, nltk, matplotlib, wordcloud
Optional ML libraries: scikit-learn, tensorflow or pytorch for advanced modeling
Tools: Jupyter Notebook for demonstration

# 🔹 Features

Scraping:
Automatically scrapes daily headlines from New Age and TBS websites.
Supports saving headlines by date.

Preprocessing:
Tokenizes, lowercases, and removes punctuation & stopwords.

Sentiment Analysis:
Classifies words into positive, negative, or neutral.
Calculates overall sentiment scores for the month.

Topic-Level Analysis:
Identifies top co-occurring words for each topic (Economy, Crime, Politics).
Filters relevant headlines and calculates topic-specific sentiment.

Prediction:
Predicts overall country sentiment: Positive, Neutral, or Negative.
Predicts topic-specific conditions.

Visualization:
Generates WordClouds for overall and topic-specific headlines.

# 📊 Example Output

Overall Condition: Neutral / Positive / Negative
Topic Conditions:
Economic: Negative
Crime: Negative
Political: Neutral

# 🚀Future Improvements

Expand scraping to more news sources
Include multi-lingual headlines (Bangla + English)
Apply machine learning models for more accurate sentiment prediction
Build a month-over-month trend dashboard for visualization

# 📧 Contact

For collaboration or feedback, feel free to reach out:
Email: tasfia.tasnim508@gmail.com
LinkedIn: https://www.linkedin.com/in/tasfiatasnim18/
