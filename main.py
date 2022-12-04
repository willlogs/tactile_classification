import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import data_reader 

data, labels = data_reader.read_data()

print(len(data))
print(len(labels))