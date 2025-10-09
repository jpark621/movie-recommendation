import pandas as pd

ratings = pd.read_csv("ratings_100k.csv")
movies = pd.read_csv("movies.csv", usecols=(0,1))
ratings = ratings.merge(movies)
ratings['titleId'] = pd.factorize(ratings['title'])[0] + 1
ratings = ratings[['userId', 'titleId']]
ratings = ratings.sample(frac=1, random_state=42)

n_users = len(set(ratings['userId'].values))
n_movies = len(set(ratings['titleId'].values))

import torch
from torch.utils.data import Dataset, DataLoader

class MovieRecommendationsDataset(Dataset):
	def __init__(self, df_X, df_y):
		self.features = torch.tensor(df_X.values, dtype=torch.int32)
		self.labels = torch.tensor(df_y.values.flatten(), dtype=torch.int64)
	def __len__(self):
		return len(self.features)
	def __getitem__(self, idx):
		return self.features[idx], self.labels[idx]

dataset_size = 100000
train_size = int(0.85 * dataset_size)
test_size = int(0.15 * dataset_size)

train_df_X = ratings[['userId']].head(train_size)
train_df_y = ratings[['titleId']].head(train_size)
train_ratings = MovieRecommendationsDataset(train_df_X, train_df_y)
train_dls = DataLoader(train_ratings, batch_size=64)

test_df_X = ratings[['userId']].tail(test_size)
test_df_y = ratings['titleId'].tail(test_size)
test_ratings = MovieRecommendationsDataset(test_df_X, test_df_y)
test_dls = DataLoader(test_ratings, batch_size=64)


from torch.nn import Module, Embedding, Linear, Softmax

class Skipgram(Module):
	def __init__(self, n_users, n_movies, n_dims):
		super(Skipgram, self).__init__()
		self.user_embeddings = Embedding(n_users, n_dims)
		self.projection = Linear(n_dims, n_movies)
		self.softmax = Softmax(dim=1)
		
	def forward(self, x):
		users = self.user_embeddings(x[:,0])
		movie_logits = self.projection(users)
		movie_probs = self.softmax(movie_logits)
		return movie_probs

n_dims = 5
user_model = Skipgram(1000, n_movies + 1, n_dims)

from torch import nn
import torch.nn.functional as F

learning_rate = 5e-3
batch_size = 64
epochs = 5

loss_fn = lambda pred, y: F.nll_loss(torch.log(pred), y)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(user_model.parameters(), lr=learning_rate)


def train_loop(dataloader, model, loss_fn, optimizer):
	size = len(dataloader.dataset)
	# Set the model to training mode - important for batch normalization and dropout layers
	# Unnecessary in this situation but added for best practices
	model.train()
	for batch, (X, y) in enumerate(dataloader):
		# Compute prediction and loss
		pred = model(X)
		loss = loss_fn(pred, y)

		# Backpropagation
		loss.backward()
		optimizer.step()
		optimizer.zero_grad()

		if batch % 100 == 0:
			loss, current = loss.item(), batch * batch_size + len(X)
			print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def test_loop(dataloader, model, loss_fn):
	# Set the model to evaluation mode - important for batch normalization and dropout layers
	# Unnecessary in this situation but added for best practices
	model.eval()
	size = len(dataloader.dataset)
	num_batches = len(dataloader)
	test_loss = 0

	# Evaluating the model with torch.no_grad() ensures that no gradients are computed during test mode
	# also serves to reduce unnecessary gradient computations and memory usage for tensors with requires_grad=True
	with torch.no_grad():
		for X, y in dataloader:
			pred = model(X)
			test_loss += loss_fn(pred, y).item()

	test_loss /= num_batches
	print(f"Test Error: \n Avg loss: {test_loss:>8f} \n")

# for t in range(epochs):
# 	print(f"Epoch {t+1}\n-------------------------------")
# 	train_loop(train_dls, user_model, loss_fn, optimizer)
# 	test_loop(test_dls, user_model, loss_fn)

#########################
# Train movie embeddings
#########################

dataset_size = 100000
train_size = int(0.85 * dataset_size)
test_size = int(0.15 * dataset_size)

train_df_X = ratings[['titleId']].head(train_size)
train_df_y = ratings[['userId']].head(train_size)
train_ratings = MovieRecommendationsDataset(train_df_X, train_df_y)
train_dls = DataLoader(train_ratings, batch_size=64)

test_df_X = ratings[['titleId']].tail(test_size)
test_df_y = ratings['userId'].tail(test_size)
test_ratings = MovieRecommendationsDataset(test_df_X, test_df_y)
test_dls = DataLoader(test_ratings, batch_size=64)

n_dims = 5
movie_model = Skipgram(n_movies + 1, 1000, n_dims)

learning_rate = 5e-3
batch_size = 64
epochs = 5

loss_fn = lambda pred, y: F.nll_loss(torch.log(pred), y)
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(movie_model.parameters(), lr=learning_rate)

# for t in range(epochs):
# 	print(f"Epoch {t+1}\n-------------------------------")
# 	train_loop(train_dls, movie_model, loss_fn, optimizer)
# 	test_loop(test_dls, movie_model, loss_fn)



