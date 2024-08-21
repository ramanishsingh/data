
import torch
from torch.utils.data import Dataset
class MyDataset(Dataset):
    def __init__(self, num_samples, num_features):
        self.data = torch.arange(num_samples * num_features).view(num_samples, num_features)
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

dataset = MyDataset(100,10)



class MockDataset(Dataset):
    def __init__(self, size):
        self.size = size
        self.data = torch.arange(size)  # Simple data that is easy to verify
    def __len__(self):
        return self.size
    def __getitem__(self, idx):
        return self.data[idx]


from torchdata.stateful_dataloader.sampler import StatefulDistributedSampler

dataloader = StatefulDistributedSampler(dataset, rank=0, num_replicas=1, shuffle=False, seed=0, drop_last=False)


print("---------- First 10 samples ----------")
for i, batch in enumerate(dataloader):
  print(i, batch)
  if i==10:
    
    break
print("-------------------------------------")

dataloader = StatefulDistributedSampler(dataset, rank=0, num_replicas=1, shuffle=False, seed=0, drop_last=False)

print("---------- First 5 samples ----------")
for i, batch in enumerate(dataloader):
  print(i, batch)
  if i==5:
    state_dict = dataloader.state_dict()
    break
print(state_dict)
dataloader.load_state_dict(state_dict)
print("---------- Next 5 samples ----------")
for i, batch in enumerate(dataloader):
  print(i, batch)
  if i==5:
    break


sampler_with_drop = StatefulDistributedSampler(dataset, num_replicas=3, rank=0, drop_last=True)
for j in sampler_with_drop:
    print(j)
sampler_without_drop = StatefulDistributedSampler(dataset, num_replicas=3, rank=0, drop_last=False)
indices_with_drop = list(iter(sampler_with_drop))
indices_without_drop = list(iter(sampler_without_drop))
print(len(indices_with_drop), len(indices_without_drop))


num_replicas = 5
all_data = []
dataset = MockDataset(100)
print(dataset[0])
for rank in range(num_replicas):
    sampler = StatefulDistributedSampler(dataset, num_replicas=num_replicas, rank=rank, shuffle=False)
    indices = list(iter(sampler))
    data_sampled = [int(dataset[i].item()) for i in indices]
    all_data.extend(data_sampled)
print(type(data_sampled[0]))
print(all_data)
#print(assertEqual(sorted([x.item() for x in all_data]), list(range(100)), "All data points should be covered exactly once across all replicas"))
