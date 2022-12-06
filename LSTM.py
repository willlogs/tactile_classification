import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import data_reader


class LSTM(nn.Module):
	def __init__(self, embedding_dim, hidden_dim, tagset_size):
		super(LSTM, self).__init__()
		self.hidden_dim = hidden_dim

		self.lstm1 = nn.LSTM(embedding_dim, hidden_dim)
		self.lstm2 = nn.LSTM(hidden_dim, hidden_dim)

		self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
		
	def forward(self, x):
		outputs, sample_size = [], x.size(1)

		h_t = torch.zeros(1, self.hidden_dim, dtype=torch.float32)
		c_t = torch.zeros(1, self.hidden_dim, dtype=torch.float32)
		h_t2 = torch.zeros(1, self.hidden_dim, dtype=torch.float32)
		c_t2 = torch.zeros(1, self.hidden_dim, dtype=torch.float32)

		for time_step in x.split(1):
			out1, (h_t, c_t) = self.lstm1(time_step, (h_t, c_t))
			out2, (h_t2, c_t2) = self.lstm2(h_t, (h_t2, c_t2))
			tag_space = self.hidden2tag(h_t2)
			tag_scores = F.log_softmax(tag_space, dim=1)
			outputs.append(tag_scores)

		return tag_scores