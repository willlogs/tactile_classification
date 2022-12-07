# imports
import sys
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

for y, d in test_data:
	idx = labels.index(y)
	print(idx)

# setup model
model = LSTM.LSTM(11, 64, 10)
loss_function = nn.CrossEntropyLoss()
lr = 0.1
optimizer = optim.SGD(model.parameters(), lr=lr)

# test accuracy before training
with torch.no_grad():
	accuracy = 0
	for y, x in test_data:
		model.zero_grad()  # clear grads
		idx = labels.index(y)
		target = [1 if idx == i else 0 for i in range(10)]
		target = torch.tensor(target)
		tag_scores = model(x)
		guess = torch.argmax(tag_scores)

		if idx == guess:
			accuracy += 1

print(f'accuracy on test data: {accuracy/len(test_data)}')

# train
for epoch in range(65):
	acc_loss = 0
	accuracy = 0
	for y, x in train_data:
		model.zero_grad()  # clear grads
		idx = labels.index(y)
		target = [1.0 if idx == i else 0.0 for i in range(10)]
		target = torch.tensor(target)
		tag_scores = model(x)
		loss = loss_function(tag_scores[0], target)
		acc_loss += loss
		loss.backward()
		optimizer.step()

		guess = torch.argmax(tag_scores)
		if idx == guess:
			accuracy += 1

		sys.stdout.write("-")
		sys.stdout.flush()

	if epoch % 15 == 0 and epoch > 0:
		lr *= 0.8
		if lr >= 0.001:
			for g in optimizer.param_groups:
				g['lr'] = lr

	print("|")
	print(f'{epoch}: {acc_loss} --- accuracy: {accuracy/len(train_data)}')

	with torch.no_grad():
		accuracy = 0
		for y, x in test_data:
			idx = labels.index(y)
			target = [1 if idx == i else 0 for i in range(10)]
			target = torch.tensor(target)
			tag_scores = model(x)
			guess = torch.argmax(tag_scores)

			if idx == guess:
				print(guess)
				accuracy += 1

	print(f'accuracy on test data: {accuracy/len(test_data)}')
