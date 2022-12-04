from torch import nn
import torch


class LSTM(nn.Module):
	def __init__(self, hidden_layers=64):
		super(LSTM, self).__init__()
		self.hidden_layers = hidden_layers
		# lstm1, lstm2, linear are all layers in the network
		self.lstm1 = nn.LSTMCell(8, self.hidden_layers)
		self.lstm2 = nn.LSTMCell(self.hidden_layers, self.hidden_layers)
		self.linear = nn.Linear(self.hidden_layers, 10)

	def forward(self, y, future_preds=0):
		outputs, num_samples = [], y.size(0)

		h_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
		c_t = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
		h_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)
		c_t2 = torch.zeros(num_samples, self.hidden_layers, dtype=torch.float32)

		for time_step in y.split(1, dim=1):
			# N, 1
			# initial hidden and cell states
			h_t, c_t = self.lstm1(time_step, (h_t, c_t))
			# new hidden and cell states
			h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
			output = self.linear(h_t2)  # output from the last FC layer
			outputs.append(output)

		for i in range(future_preds):
			# this only generates future predictions if we pass in future_preds>0
			# mirrors the code above, using last output/prediction as input
			h_t, c_t = self.lstm1(output, (h_t, c_t))
			h_t2, c_t2 = self.lstm2(h_t, (h_t2, c_t2))
			output = self.linear(h_t2)
			outputs.append(output)

		# transform list to tensor
		outputs = torch.cat(outputs, dim=1)
		return outputs
