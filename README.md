# stock-cluster-nlp

September 2017

This is the second post in a series on using Machine Learning in pairs trading. Pairs trading is perhaps the earliest form of relative value quantitative trading in equities. This series attempts to bring to bear some modern Machine Learning tools to the pair trading investment process. In the first post of this series, I used `DBSCAN` clustering on latent statistical factors in price returns, the Morningstar `financial_health_grade`, and each company's market capitalization to find stocks which had a high likelihood of being valid eligible pairs in a pairs trading strategy. In this post, I take a very different path driven by the following question:

_Is it possible to find valid eligible pairs without using any price data at all?_

If we could do that, perhaps the process would be highly robust. From first principles, why do certain stocks have highly related price series (i.e., why could valid pairs exist)? I conjecture that it is because certain stocks:

- operate in similar business lines
- have similar economic exposures
- have similar regulatory burdens
- have a coincident set of homogenous investors
- operate in the same geographic markets

Therefore, if we could read about and understand the business of each company and then link up companies based on this understanding, we should have a robust set of potential eligible pairs. Human analysts are good at this kind of task, but can a machine do as well if not better? Well, this is a perfect task for Machine Learning, and, specifically, the sub-field of Natural Language Processing.

In this post, I:

- gather business profiles on a couple thousand stocks,
- utilize the `scikit-learn` natural language processing functionality `CountVectorizer` and `TfidfTransformer` to "read" these descriptions and extract important and novel concept features across all companies,
- cluster stocks, again with `DBSCAN`, to find stocks that have similar profiles,
- visualize the features for a handful of stocks via `WordCloud` to get some intuition on what the ML model is learning, and
- lastly, inspect the time series of discovered clusters to see if this process, having no stock price series inputs at all, are related.
