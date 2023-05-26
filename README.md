## Online Media Brand-Analysis using BERTopic 
BERTopic model has been applied to texts taken out from 3 different sources(blogs, news and social media) to find out whether certain common topics alongside of negative viral social media posts are mentioned about a specific airline.

### Exploratory Data Analysis
Three data sources(blog, news and social data) have been read to dataframes from parquet format. For each of these sources, aside of looking at the data summaries like number of rows, language, author and domain distributions and publish date ranges, the main topics mentioned in each of the sources also summarized before getting into the details of the specific topics for the airline(s). The following topic modeling solution has been proposed for this purpose and used for the analysis made in the rest of this study.

### Proposed Solution
In order to focus on a specific airline for the sake of this study, a wordcloud analysis is made on a sample of the social media data to see which airlines mentioned most frequently.

![alt text](/images/wordcloud.png) 

As can be seen, the main airline that comes out happened to be "United Airlines" among some other less mentioned ones like Southwest, Spirit and American airlines. So, for the topic modeling part, "United Airlines" has been the main focus.

The each data sources has been cleaned, filtered out and passed into the topic model by the following framework:
- Empty and very short texts/bodies have been filtered out, tweets have been cleaned using a custom function and all the characters in the texts have been lowered. 
```python
if media_type == 'social':
    df = df[(df.language == 'en') & (df.text.notna())]
    df['text_clean'] = df['text'].apply(lambda x: clean_tweet(x))
else:
    df = df[(df['body'].str.len() > 30) &
            (df.language == 'en') &
            (df.body.notna())].reset_index()
    df['text_clean'] = df['body'].apply(lambda x: x.lower())
```

- Only the rows(entries) containing "united airlines" have been subsetted and the dataframe converted to a list to be processed by the selected topic modeling.
```python
df = df[df['text_clean'].str.contains("united airlines")]
text_list = df['text_clean'].tolist()
```

- Finally, the text samples from each three main data have been pased to the BERTopic model to extract the main topics based on the keywords frequency in each of them. BERTopic configuration was kept simple for the sake of this study and a very generic BERTopic model with stopwords CountVectorizer has been fitted to the corresponding corpuses.    
```python
vectorizer_model = CountVectorizer(stop_words="english")
united_model = BERTopic(
    vectorizer_model=vectorizer_model, 
    language='english', 
    calculate_probabilities=True,
    verbose=True)
news_topics, news_probs = united_model.fit_transform(text_list)    
```

#### Blog Data
The relevant blog data includes around 100 blog post about mostly generic topics and did not seem promising to understand much about what's menitoned on "United Airlines" or its competitors.

![alt text](/images/blog_summary.png)


#### News Data
The relevant news data includes around 3000 news mentioning "United Airlines", and the most prominent topics can be seen below with the corresponding keywords.

![Alt text](/images/news_summary.png)

When looking into the word frequencies in some of them, and reading a few samples labeled with those topics, one can see that some news mention possibly flight destinations while some others a bad experience happen to a celebrity during a flight which can be considered as a negative campaign. The details will be discussed in the results section.      

#### Social Data
The relevant social data includes around 10000 social media entries(by twitter and reddit) mentioning "United Airlines", and the most prominent topics can be seen below with the corresponding keywords.

![alt text](/images/social_summary.png)

Looking into the main topics extracted from the social media data, a similar set of topics to the news data data stand out. With a quick look, we understand that possibly an incident happen to a celebrity, and most other posts are about more generic industry related news.
The details of these will be discussed in the results section.      


### The Results 

#### Negative viral online media topics on the airline


#### Common topics mentioned on the airline


#### General topics & trends mentioned in the industry and within its competitor


#### Latent opportunites or risks that may be addressed proactively 



## About the github repo
This repo has been built with OOO fashion with the below structured.
... tree of the repo...
The way it's designed allow to pip install it repo as a wholesome package.

Once the repo has been cloned and a virutal environment created, the main python file can be run for any desired airlines following the format here: 



## Further Study

An automatic query structured will be implemented for a specific keyword or phrase to extract the common topics mentioned within a certain time period with the following structure:

- Online media sources will be searched with the specifi keyword and time period
- Most frequent topics will be brought 
- Sentiment Analysis will be made 

