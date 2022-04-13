[![scrape_process_analyse](https://github.com/edgar-hd/UN-tweets/actions/workflows/pipeline_scrape_process_analyse.yml/badge.svg)](https://github.com/edgar-hd/UN-tweets/actions/workflows/pipeline_scrape_process_analyse.yml)

# Analysis of tweets involving the hashtag "#UN"

For identifying current trends and making novel predictions on how long trends in twitter will last, this repository runs daily and commits to itself. You can find the resulting list of current trends and predictions below.

Please refer to the jupyter notebook [proj1_analyse_tweets.ipynb](proj1_analyse_tweets.ipynb) for a breakdown of the project, I write below the motivation and summary which can also be found in the notebook.

## Motivation

NOTE: Motivation written on 2nd of March 2022.

In recent days I have found my twitter feed replete of news regarding the recent events taking place in Ukraine. While I follow some news outlets, most of the accounts I follow are either people I know or related to learning Japanese language. This trend made me curious as to how much twitter can be used to track world political events, and whether I could discover past political events using just twitter. I simplistically thought of which keyword could relate to such events and decided upon "#UN" relating to the United Nations. In principle this organisation was established to serve as a forum for discussion between nations and thus it would be reasonable that twitter users would reference the UN during major political world events (Official website: https://www.un.org/en/).

I decided to work with a moderate dataset covering all tweets using "#UN" since January 2021 and see what results I could extract from there. Indeed as can be seen by the end of my analysis, it may now be sensible to include tweets tagging the user "@UN" or extending the time frame of my analysis. Nonetheless, as an initial pilot I am happy with the results.

It is important to acknowledge that as analysis are made by humans, they will fundamentally be biased. I have made my greatest effort to be as unbiased as possible in my analysis and writing, while understanding that removing all bias is likely impossible. This may result in a somewhat dispassionate tone regarding major events which bring great suffering to many people, yet it is not the purpose of this project to bring about a political message and for that reason I find this approach the most appropriate.

## Exemplifying result of processed, analysed and clustered data
A Dendogram of major trends, using cosine similarity to group keywords according to similarity in their temporal occurence.

![dendogram_trends](https://user-images.githubusercontent.com/43865617/159815652-d5561e96-6010-4f26-a24f-e85175dec8dd.png)

## Summary <a class="anchor" id="summary"></a>
In summary I have analysed tweets ranging from the 1st of January 2021 to the 24th of February 2022 containing the #UN in English. Through a combination of statistics, deep learning, dimensionality reduction and clustering I have been able to distill the main talking points for twitter users as well as their sentiment in regards to these topics. Overall I have found that there is a substantial response to outbreaks of conflict, in particular those that receive extensive media coverage. I find 4 "large" trends that encompass many different subtopics and which received most of the attention of Twitter users, and 13 trends overall which had varying levels of response but all had clear peaks at specific timepoints.

Note the above numbers have likely changed as time passes but I will not change them with every update.

I have further trained a random forest model to predict how long ongoing trends will remain actively tweeted about. This model and prediction are updated as data is collected daily. The below plot containing ongoing trends and their predictions of duration is therefore also updated daily.

![dendogram_trends](figures/fig4d_current_topics_lifetime.pdf)

Further avenues of research are to test these approaches on longer timescales to try to identify cyclic patterns, some being obvious such as relating to the opening of the general assembly each year; this may also identify the frequency of "trending" events related to the UN. Another interesting approach would be to try with different hashtags as controls, ideally one completely unrelated dataset and one closely related dataset to serve as negative and positive controls respectively. Overall I find this analysis useful to understand user response to political events and the results suggests this is a valid pipeline for extracting information regarding other topics on social media.
