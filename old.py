import numpy as np
import torch, torch.nn as nn
from joblib import Parallel, delayed

N, n = 2 ** 8, 5

def make_random_graphs(batch_size, p_edge):
  graphs = torch.zeros(batch_size, n, n)
  graphs[torch.rand(batch_size, n, n) < p_edge] = 1
  tmp = graphs.clone()
  for i in range(int(np.log2(n) + 1)):
    tmp = torch.minimum(tmp @ tmp, torch.ones_like(tmp))
  answers = tmp[:, 0, n - 1]
  return graphs, answers

def find_p_edge():
  l, r = 0, 1
  while r - l > 1e-6:
    m = (l + r) / 2
    batch_size = 256
    graphs, answers = make_random_graphs(batch_size, m)
    if torch.sum(answers) / batch_size < 0.5:
      l = m
    else:
      r = m
  return l

p_edge = find_p_edge()
print(f'{N=}, {n=}, {p_edge=}')

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.Q = torch.randint(n, size=(N, 2))
    self.logits_if_no = nn.Parameter(torch.randn(N, N))
    self.logits_if_yes = nn.Parameter(torch.randn(N, N))
  def forward(self, graphs):  # returns probabilities of answering no and yes
    answers = graphs[:, self.Q[:, 0], self.Q[:, 1]]
    ps_if_graphs = self.logits_if_no.repeat(len(graphs), 1, 1)
    mask = (answers == 1).unsqueeze(-1).repeat(1, 1, N)
    ps_if_graphs[mask] = self.logits_if_yes.unsqueeze(0).repeat(len(graphs), 1, 1)[mask]
    ps_if_graphs = nn.functional.softmax(ps_if_graphs, -1)
    state = torch.zeros(len(graphs), 1, N, device=self.logits_if_no.data.device)
    state[:, 0, 0] = 1
    for t in range(N):
      state = state @ ps_if_graphs
    return state[:, 0, -2], state[:, 0, -1]

device = 'cuda'
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e+1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.7)
batch_size = 64
for epoch in range(1, 61):
  ttl_loss = 0
  batches_per_epoch = 100
  for i in range(batches_per_epoch):
    graphs, answers = make_random_graphs(batch_size, p_edge)
    graphs, answers = graphs.to(device), answers.to(device)
    ps_no, ps_yes = model(graphs)
    loss = torch.mean((ps_yes - answers) ** 2 + (ps_no - (1 - answers)) ** 2)
    ttl_loss += loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  scheduler.step()
  print('epoch', epoch, 'lr', scheduler.get_last_lr(), 'avg loss', ttl_loss / batches_per_epoch)

print(f'questions:')
print(model.Q)
print(f'if no:')
print(model.logits_if_no.argmax(-1))
print(f'if yes:')
print(model.logits_if_yes.argmax(-1))
