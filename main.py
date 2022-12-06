# imports
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import data_reader
import LSTM

from random import shuffle

# global vars
test_proportion = 0.2

# read dataset
print("reading dataset...")
data, labels = data_reader.read_data()
print(f"Number of data sequences: {len(data)}")
print(f"Number of Classes: {len(labels)}")

# divide into train/test
print("Dividing to test/train sets")

zipped_data = []
for idx, dp in enumerate(data):
	zipped_data.append((labels[idx//10], torch.tensor(dp)))

shuffle(zipped_data)

divison_idx = int(test_proportion * len(zipped_data))
test_data = zipped_data[0:divison_idx]
train_data = zipped_data[divison_idx:]
print(f"number of test dps: {len(test_data)}")
print(f"number of train dps: {len(train_data)}")

# setup model
model = LSTM.LSTM(8, 64, 10)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01)

# train
for epoch in range(10):
	acc_loss = 0
	for y, x in train_data:
		model.zero_grad() # clear grads
		idx = labels.index(y)
		target = [1 if idx == i else 0 for i in range(10)]
		target = torch.tensor(target)
		tag_scores = model(x)
		loss = loss_function(tag_scores[0], target)
		acc_loss += loss
		loss.backward()
		optimizer.step()
		print('-', end='')

	print("---")
	print(f'{epoch}: {acc_loss}')

with torch.no_grad():
	accuracy = 0
	for y, x in test_data:
		model.zero_grad() # clear grads
		idx = labels.index(y)
		target = [1 if idx == i else 0 for i in range(10)]
		target = torch.tensor(target)
		tag_scores = model(x)
		guess = torch.argmax(tag_scores)

		if idx == guess:
			accuracy += 1

print(f'accuracy on test data: {accuracy/len(test_data)}')