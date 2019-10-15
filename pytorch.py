import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autog

x_data = np.array([[0, 0], [0, 1], [1, 0]])
y_data = np.array([[0, 1, 1]]).T
x_data = autog.Variable(torch.FloatTensor(x_data))
y_data = autog.Variable(torch.FloatTensor(y_data), requires_grad=False)

in_dim = 2
out_dim = 1
epochs = 15000
epoch_print = epochs / 5
l_rate = 0.001

class NeuralNet(nn.Module):
  def __init__(self, input_size, output_size):
    super(NeuralNet, self).__init__()
    self.lin1 = nn.Linear(input_size, output_size)

  def forward(self, x):
    return self.lin1(x)

model = NeuralNet(in_dim, out_dim)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.Adam(model.parameters(), lr=l_rate)

# Accuracy
for epoch in range(epochs):
  optimizer.zero_grad()
  pred = model(data_x)
  loss = criterion(pred, data_y)
  loss.backward()
  optimizer.step()
  if (epoch + 1) % epoch_print == 0:
    print("Epoch %d  Loss %.3f" %(epoch + 1, loss.item()))

# Prediction
for x, y in zip(data_x, data_y):
  pred = model(x)
  print("Input", list(map(int, x)), "Pred", int(pred > 0), "Output", int(y))
