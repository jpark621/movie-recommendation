# movie-recommendation
Adding contextual features to a movie recommendation system

# Notes
## 10.07.25
 * Implemented Collaborative filtering to 100k [link](https://www.kaggle.com/code/jhoward/collaborative-filtering-deep-dive/notebook)
 * sigmoid_range and Adam have highest gains
   * sigmoid_range forces predictions to fit a sigmoid function in range \[0, 5\] (technically 5.5 for empirical reasons).
   * Adam is a simple optimizer change.
