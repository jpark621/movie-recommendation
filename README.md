# movie-recommendation
Adding contextual features to a movie recommendation system

# Contextual features
Skip-gram model is a technique to generate word embeddings by predicting context *words* from a given target word. Instead, we can use user embeddings to predict a *context*, in this case, the movies that user has watched. We do the same for movie embeddings, where given a movie, we predict its audience. Each embedding should capture the semantic meaning, that is, similar users are near similar users and similar movies are with similar movies.

The architecture is as follows:
  for movie_j in context j:  <-- do this in data, not modeling.
    Embedding, Linear (user embedding to all movies vector), softmax (casts vectors to movie prob)

The loss compares the movie softmax prob with the actual movie (whether the user has watched the movie or not):

     F.nll(log(movie_prob), target movie)

This is equivalent to cross-entropy on the all-movies vector (logits).

If user a has seen a, b, c movies, we will make a loss for each target movie. Each data sample will look like

	userId, titleId (user a, movie a)
	userId, titleId (user a, movie b)
	userId, titleId (user a, movie c)


# Notes
## 10.07.25
 * Implemented Collaborative filtering to 100k ([link](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive/notebook))
 * sigmoid_range and Adam have highest gains
   * sigmoid_range forces predictions to fit a sigmoid function in range \[0, 5\] (technically 5.5 for empirical reasons).
   * Adam is a simple optimizer change.
