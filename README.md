## M-HiTS: Multilingual Hindi Text 

Hindi Text Summarizer for News Articles in Hindi Language
Hindi Text Summarizer is a summarizer generating tool, written in python. It is developed for everyone where they can come up with the summary in short for long news articles in Hindi

**Installation Guidelines**
 
 * [Python Official Website](https://www.python.org/)
 * [Anaconda](https://www.continuum.io/downloads)
 
 **Data**
 * We have scrapped the news articles using [beautiful soup](https://pypi.python.org/pypi/beautifulsoup4) library in python from this 	[website](http://www.sampadkiya.com/) 

| Features | Values |
| --- | --- |
| Number Of Articles|      4,316   |
|   Sentence Count  |    1,52,270  |
|     Word Count    |    30,88,571 |

* We have generated the summary of the above Hindi articles using this [tool](https://bigdatasummarizer.com/summarizer/online/advanced.jsp?ui.lang=en)

| Features | Values |
| --- | --- |
| Number Of Articles |     4,316   |
|   Sentence Count   |    63,915   |
|    Word Count      |   17,18,785 |


**Features**

* Topic Feature
* Sentence Position Feature
* Proper Noun Feature
* Cue Word Feature
* Bigram Feature
* Unknown Word Feature

**Testing Algorithms**

* Gradient Boost
* Random Forest
* AdaBoost
* Svm
* K-Nearest Neighbor
* Extremely Randomized Trees

**Testing**

* Bleu Score
