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

### Negative viral online media topics on the airline
Focusing on the United Airlines, the main viral negative social media posts have been about an incident where a famous baseball player's pregnant wife had to clean up popcorn in the cabin. The topic came out several times both in the news data and social data with different topic clusters. When we look into some of the posts labeled by this topic we see similar posts about the incident:
- > 'blue jays anthony bass slams united airlines for making pregnant wife clean kids messes on hands and knees'

- > 'a united airlines flight attendant allegedly forced the 22 week pregnant wife of toronto blue jays pitcher anthony bass to clean the planes cabin on her hands and knees' 

### Common topics mentioned on the airline
Other common topics mentioned the airline have been more generic news and posts about new flight destinations, quarterly earnings etc.:
- > 'united airlines will fly to 114 different international cities this summer and has expanded its flying by 25 versus last year' 
- > 'united airlines stock chart fibonacci analysis 042923 trading idea entry point 44 61 80'

### General topics & trends mentioned in the industry and within its competitor

Some interesting topics and trends mentioned in the industry include a proposal for free seats for plus size travelers, Southwest nationwide grounding issues and mixed customer service experiences with Delta among others.

```
Topic	Count	Name
-1	    6852	-1_ng_plane_airline_nh
 0	    401	    0_delta_feet_spotted_flight
 1	    382	    1_spirit_airlines_ghosts_fly
 2	    340	    2_told_didnt_relationship_friends
 3	    313	    3_spirit_feet_spotted_flight
 4	    283	    4_level_body_astral_world
 5	    264	    5_united_airlines_rt_express
 6	    256	    6_airline_airways_cambrian_ser
 7	    229	    7_delta_deltas_customer_service
 8	    200	    8_southwest_airlines_texas_hsr
 9	    184	    9_seats_seat_plus_size

Free Seats(Topic_9) Keywords:
[('seats', 0.07435451872439745),
 ('seat', 0.04311568214160679),
 ('plus', 0.030041212453550948),
 ('size', 0.028543438746437597),
 ('demands', 0.02683698959894252),
 ('fat', 0.024430223163090247),
 ('petition', 0.021261785945257405),
 ('bathrooms', 0.019750317095925285),
 ('sized', 0.018882864139952184),
 ('free', 0.018602459436824727)]
 
 Delta(Topic_7) Keywords:
 [('delta', 0.0936722479712925),
 ('deltas', 0.023920515464044147),
 ('customer', 0.01671281097848389),
 ('service', 0.012589097178904348),
 ('book', 0.012288369143513337),
 ('booking', 0.012153269979377232),
 ('lines', 0.011798737696350863),
 ('airlines', 0.01032338025151275),
 ('offers', 0.009779246237443782),
 ('reservation', 0.009517391824111586)]
 ```
- >  'a fat tiktoker is demanding that the faa protect plus size travelers by providing obese people with free airplane seats hit the obese tiktoker wants free airline seats for fatties s1 e66'
- > 'daily reminder if you fly on delta airlines youre doing so at great risk my recommendation if youre planning a trip use another airline youll thank me later if you cant book with another airline just walk youll arrive at your destination sooner much safer'

- > 'Southwest passengers face delays after nationwide grounding - WDET 101.9 FM'


## About the github repo
The repo needs a few more touches to be finished with the inference pipeline. 
It has been built with OOP fashion with the below main structure.
```
├── notebooks
│   └── 1_Analysis.ipynb
├── requirements.txt
├── setup.py
└── src
    ├── __init__.py
    ├── components
    │   ├── __init__.py
    │   ├── data_ingestion.py
    │   ├── data_transformation.py
    │   └── model_trainer.py
    ├── exception.py
    ├── logger.py
    ├── pipeline
    │   ├── __init__.py
    │   └── inference.py
    └── utilities.py
 ```   
Once it's done, it will allow to run a real-time inference pipeline through pip install.



## Further Study

An automatic query structured will be implemented for a specific keyword or phrase to extract the common topics mentioned within a certain time period with the following structure:

- Online media sources will be searched with the specific keyword and time period
- Most frequent topics will be brought 
- Sentiment Analysis will be made 

