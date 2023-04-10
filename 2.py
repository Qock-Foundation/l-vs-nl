import numpy as np
import matplotlib.pyplot as plt
import torch, torch.nn as nn
from joblib import Parallel, delayed

N = 2048
T = 128
n = 6

def make_random_graphs(batch_size, p_edge):
  graphs = torch.zeros(batch_size, n, n)
  graphs[torch.rand(batch_size, n, n) < p_edge] = 1
  tmp = graphs.clone()
  for i in range(int(np.log2(n) + 1)):
    tmp = torch.minimum(tmp @ tmp, torch.ones_like(tmp))
  answers = tmp[:, 0, n - 1].type(torch.bool)
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

def logprodexp(a, b):
  '''
    shape of a: (m, n) or (b, m, n)
    shape of b: (n, k) or (b, n, k)
  '''
  #return torch.log(torch.exp(a) @ torch.exp(b))
  m, n, k = *a.shape[-2:], b.shape[-1]
  batched = a.ndim == 3
  a = a.unsqueeze(-1).repeat((1, 1, 1, k) if batched else (1, 1, k))
  b = b.unsqueeze(-3).repeat((1, m, 1, 1) if batched else (m, 1, 1))
  return torch.logsumexp(a + b, -2)

test_a = torch.randn(3, 5)
test_b = torch.randn(5, 2)
assert torch.allclose(torch.exp(test_a) @ torch.exp(test_b), torch.exp(logprodexp(test_a, test_b)))
test_a = torch.randn(7, 6, 4)
test_b = torch.randn(7, 4, 7)
assert torch.allclose(torch.exp(test_a) @ torch.exp(test_b), torch.exp(logprodexp(test_a, test_b)))

class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.Q = torch.randint(n, size=(N, 2))
    self.logits_if_no = nn.Parameter(torch.randn(N, N))
    self.logits_if_yes = nn.Parameter(torch.randn(N, N))
  def forward(self, graphs, Temp=1):  # returns probabilities of answering no and yes
    answers = graphs[:, self.Q[:, 0], self.Q[:, 1]]
    log_ps_if_graphs = self.logits_if_no.repeat(len(graphs), 1, 1) / Temp
    mask = (answers == 1).unsqueeze(-1).repeat(1, 1, N)
    log_ps_if_graphs[mask] = self.logits_if_yes.unsqueeze(0).repeat(len(graphs), 1, 1)[mask] / Temp
    log_ps_if_graphs = nn.functional.log_softmax(log_ps_if_graphs, -1)
    state = torch.full((len(graphs), 1, N), float('-inf'), device=self.logits_if_no.data.device)
    state[:, 0, 0] = 0
    for t in range(T):
      state = logprodexp(state, log_ps_if_graphs)
    entropies = -(torch.exp(state) * state).sum(-1)
    return state[:, 0, -2], state[:, 0, -1], entropies
  def fix(self):
    with torch.no_grad():
      self.logits_if_no[-2] = float('-inf')
      self.logits_if_no[-2, -2] = 0
      self.logits_if_yes[-1] = float('-inf')
      self.logits_if_yes[-1, -1] = 0

device = 'cuda'
model = Model().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-1)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
batch_size = 8
for epoch in range(1, 41):
  Temp = 1
  ttl_correctness_loss, ttl_entropy_loss = 0, 0
  batches_per_epoch = 100
  for i in range(batches_per_epoch):
    graphs, answers = make_random_graphs(batch_size, p_edge)
    graphs, answers = graphs.to(device), answers.to(device)
    ps_no, ps_yes, entropies = model(graphs, Temp)
    correctness_loss = -(torch.sum(ps_yes[answers]) + torch.sum(ps_no[torch.logical_not(answers)])) / batch_size
    ttl_correctness_loss += correctness_loss.item()
    entropy_loss = -torch.mean(entropies)
    ttl_entropy_loss += entropy_loss.item()
    k = 0.02
    loss = correctness_loss + k * entropy_loss
    optimizer.zero_grad()
    loss.backward()
    if np.random.random() < 0.3 / batches_per_epoch:
      print('typical grad', torch.mean(torch.abs(model.logits_if_no.grad)), torch.mean(torch.abs(model.logits_if_yes.grad)))
    optimizer.step()
    model.fix()
  scheduler.step()
  print('epoch', epoch, 'lr', scheduler.get_last_lr(), 'avg correctness loss', ttl_correctness_loss / batches_per_epoch, 'avg entropy loss', ttl_entropy_loss / batches_per_epoch)
#fig, axes = plt.subplots(1, 2, figsize=(14, 7))
#m3, p3 = torch.full(model.logits_if_no.shape, -3), torch.full(model.logits_if_yes.shape, +3)
#axes[0].imshow(torch.minimum(torch.maximum(model.logits_if_no.cpu().detach(), m3), p3))
#axes[1].imshow(torch.minimum(torch.maximum(model.logits_if_yes.cpu().detach(), m3), p3))
#plt.show()

print()
while True:
  state = 0
  while True:
    if state == N - 2:
      print('ANSWER NO')
      print()
      break
    if state == N - 1:
      print('ANSWER YES')
      print()
      break
    print('state', state, 'question', model.Q[state][0].item(), model.Q[state][1].item(), 'answer: ', end='')
    answer = input()
    if answer == 'yes':
      state = model.logits_if_yes[state].argmax(-1).item()
    elif answer == 'no':
      state = model.logits_if_no[state].argmax(-1).item()
    else:
      print()
      break
