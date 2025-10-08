import pandas as pd
import torch

ratings = pd.read_csv("ratings_100k.csv")
movies = pd.read_csv("movies.csv", usecols=(0,1))
ratings = ratings.merge(movies)
ratings['titleId'] = pd.factorize(ratings['title'])[0] + 1
ratings = ratings[['userId', 'titleId', 'rating']]
ratings = ratings.sample(frac=1, random_state=42)

n_users = len(set(ratings['userId'].values))
n_movies = len(set(ratings['titleId'].values))

from torch.utils.data import Dataset, DataLoader

class MovieRecommendationsDataset(Dataset):
	def __init__(self, df_X, df_y):
		self.features = torch.tensor(df_X.values, dtype=torch.int32)
		self.labels = torch.tensor(df_y.values.flatten(), dtype=torch.float32)
	def __len__(self):
		return len(self.features)
	def __getitem__(self, idx):
		return self.features[idx], self.labels[idx]

dataset_size = 100000
train_size = int(0.85 * dataset_size)
test_size = int(0.15 * dataset_size)

train_df_X = ratings[['userId', 'titleId']].head(train_size)
train_df_y = ratings[['rating']].head(train_size)
train_ratings = MovieRecommendationsDataset(train_df_X, train_df_y)
train_dls = DataLoader(train_ratings, batch_size=64)

test_df_X = ratings[['userId', 'titleId']].tail(test_size)
test_df_y = ratings['rating'].tail(test_size)
test_ratings = MovieRecommendationsDataset(test_df_X, test_df_y)
test_dls = DataLoader(test_ratings, batch_size=64)

from torch.nn import Module, Embedding

def sigmoid_range(x, low, high):
	"Sigmoid function with range `(low, high)`"
	return torch.sigmoid(x) * (high - low) + low

class DotProduct(Module):
	def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
		super(DotProduct, self).__init__()
		self.user_factors = Embedding(n_users, n_factors)
		self.movie_factors = Embedding(n_movies, n_factors)
		self.y_range = y_range
		
	def forward(self, x):
		users = self.user_factors(x[:,0])
		movies = self.movie_factors(x[:,1])
		return sigmoid_range((users * movies).sum(dim=1), *self.y_range)

class DotProductBias(Module):
	def __init__(self, n_users, n_movies, n_factors, y_range=(0,5.5)):
		super(DotProductBias, self).__init__()
		self.user_factors = Embedding(n_users, n_factors)
		self.user_bias = Embedding(n_users, 1)
		self.movie_factors = Embedding(n_movies, n_factors)
		self.movie_bias = Embedding(n_movies, 1)
		self.y_range = y_range
		
	def forward(self, x):
		users = self.user_factors(x[:,0])
		movies = self.movie_factors(x[:,1])
		res = (users * movies).sum(dim=1, keepdim=True)
		res += self.user_bias(x[:,0]) + self.movie_bias(x[:,1])
		return sigmoid_range(res, *self.y_range)

n_factors = 5
# model = DotProduct(1000, n_movies + 1, n_factors)
model = DotProductBias(1000, n_movies + 1, n_factors)

from torch import nn

learning_rate = 5e-3
batch_size = 64
epochs = 5

loss_fn = nn.MSELoss()
# optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

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

for t in range(epochs):
	print(f"Epoch {t+1}\n-------------------------------")
	train_loop(train_dls, model, loss_fn, optimizer)
	test_loop(test_dls, model, loss_fn)
