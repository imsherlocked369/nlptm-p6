# TODO
## General
- [ ] Find poems from Gutenberg
	- Poem collections or epic poems ?
	- Divine Comedy - Dante Alighieri
	- Paradise Lost - John Milton
	- The Lamb and The Tiger - William Blake

## Task 1-3
- [ ] Determine TF-IDF of tokens in poem
- [ ] Estimate how TF-IDF score of stopwords compare to non-stopwords tokens score
- [ ] It is commonly acknowledged that TF-IDF of stopwords should be lower
	- Is the claim valid, test using statistics
	- Find TF-IDF for stopwords in normal text
		- Brown corpus ?
		- Gutenberg non poem texts ?
	- Poem are not necessarily comparably to the general texts
		- They are usually shorter
			- Epic poems are not
		- They may use untypical sentence structures
			- They may omit stopwords etc
		- Stopwords may play more important role
- Only having $n$ many documents is going cause $IDF(q)$ to only have $n+1$ possible values
	- This reduces the validity of using tf-idf as effective measure for stopword selection
- Typical stopwords list are not good fit for poems
	-  

BM25 for query
$$
score(D,Q) = \sum_{i=1}^nIDF(q_i) \cdot \frac{f(q_i,D) \cdot (k_1+1)}{f(q_i,D) + k_1 \cdot (1-b+b \cdot\frac{|D|}{avgdl})}
$$
BM25 for single token
$$
score(D,q) = IDF(q) \cdot \frac{f(q,D) \cdot (k_1+1)}{f(q,D) + k_1 \cdot (1-b+b \cdot\frac{|D|}{avgdl})}
$$

Function $f(q,D)$ is the count of token $q$ in document $D$.
Variable $avgdl$ is the average document length in the corpus.
Variable $k_1 \in [1.2\;2.0]$ and $b=0.75$