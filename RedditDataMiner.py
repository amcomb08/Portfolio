import praw
import pandas as pd
from textblob import TextBlob
import matplotlib.pyplot as plt

########### Replace these values with your own to connect to reddit API ##################
client_id = "kpKjhPzJAk39N7qjlF4-3Q"
client_secret = "fl-7akBW8UML9g13tx1N8rCIZQy_Hw"
user_agent = "MSW:CSE532Anthony:v0.1.0 (by u/ShiftedCube)"

reddit = praw.Reddit(
    client_id=client_id,
    client_secret=client_secret,
    user_agent=user_agent,
)

subreddits = ["NBA", "nfl", "soccer", "hockey", "baseball"]

# DataFrame to store data
data = {
    "subreddit": [],
    "title": [],
    "sentiment_polarity": [],
    "submission_type": [],
}

for subreddit_name in subreddits:
    subreddit = reddit.subreddit(subreddit_name)
    
    # Get the top submissions
    top_submissions = subreddit.top(limit=20)
    for submission in top_submissions:
        sentiment = TextBlob(submission.title).sentiment
        data["subreddit"].append(subreddit_name)
        data["title"].append(submission.title)
        data["sentiment_polarity"].append(sentiment.polarity)
        data["submission_type"].append('top')

    # Get the most controversial submissions
    controversial_submissions = subreddit.controversial(limit=20)
    for submission in controversial_submissions:
        sentiment = TextBlob(submission.title).sentiment
        data["subreddit"].append(subreddit_name)
        data["title"].append(submission.title)
        data["sentiment_polarity"].append(sentiment.polarity)
        data["submission_type"].append('controversial')

# Create a DataFrame from the data dictionary
df = pd.DataFrame(data)
fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
fig.suptitle("Sentiment Polarity of Top and Most Controversial Submissions")

# Plot the sentiment polarity of top submissions
top_sentiment_polarity = [df[(df['subreddit'] == subreddit) & (df['submission_type'] == 'top')]['sentiment_polarity'] for subreddit in subreddits]
axes[0].boxplot(top_sentiment_polarity, labels=subreddits)
axes[0].set_title("Top Submissions")
axes[0].set_xlabel("Subreddits")
axes[0].set_ylabel("Sentiment Polarity")
axes[0].tick_params(axis='x', rotation=45)

controversial_sentiment_polarity = [df[(df['subreddit'] == subreddit) & (df['submission_type'] == 'controversial')]['sentiment_polarity'] for subreddit in subreddits]
axes[1].boxplot(controversial_sentiment_polarity, labels=subreddits)
axes[1].set_title("Controversial Submissions")
axes[1].set_xlabel("Subreddits")
axes[1].set_ylabel("Sentiment Polarity")
axes[1].tick_params(axis='x', rotation=45)

plt.show()
