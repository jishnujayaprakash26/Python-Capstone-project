# Programming with Python: Capstone - Social Media Analytics

In this project, we analyzed a dataset that contains social media messages
(tweets and facebook posts) made by United States politicians in 2013 over a
two-day period. We parsed the labels and text of the messages to add
additional data on hashtag use, message sentiment, and poster identification,
which can be used to compare politicians across several vectors.

In the first section, we organized the data by adding several additional columns
to the dataframe, to include information on name, location, position, sentiment,
and hashtags by parsing some of the text included in the original CSV. This
reformatting made analysis easier in the following stage.

In the second section, we dived deeper into actually analyzing the social media
messages in the CSV. We have filtered data by information about the poster
and did different analyses on hashtags and sentiments to compare the positive
and negative feelings on different topics between different politicians. We also
explored which hashtags are most popular and what their average sentiments
are.

### Note:
To keep the starter file from becoming too crowded, all the tests have
been moved to the file social_tests.py. The functions in this file are called at
the bottom of social.py.

##### Data Source Used: kaggle
### Components & Libraries of Python used:
Library: This project uses the pandas library to perform data analysis.

Core Components: Data, Variables, Expressions, Statements

Control Flow of Program: Functions, Conditionals, Loops, Recursion, Classes

Organization of data into new structures: Lists, Sets, Dictionaries, Objects
