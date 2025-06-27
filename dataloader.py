import numpy as np
import torch
import torch.utils.data
import torch.nn.functional as F
class EpisodeSampler:
    def __init__(self, label, n_batch, n_cls, n_per, fix_seed=True):
        self.n_batch = n_batch
        self.n_cls = n_cls
        self.n_per = n_per
        self.fix_seed = fix_seed

        label = np.array(label)
        self.m_ind = []
        for i in range(max(label) + 1):
            ind = np.argwhere(label == i).reshape(-1)
            ind = torch.from_numpy(ind)
            self.m_ind.append(ind)
        if self.fix_seed:
            np.random.seed(42)
            self.cached_batches = []
            for i in range(self.n_batch):
                batch = []
                classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                for c in classes:
                    l = self.m_ind[c]
                    if len(l) >= n_per:
                        pos = np.random.choice(range(len(l)), self.n_per, False)
                        batch.append(l[pos])
                    elif (len(l)>0 and len(l) < n_per):
                        pos = np.random.choice(range(len(l)), self.n_per, True)
                        batch.append(l[pos])
                        
                if batch:
                    batch = torch.stack(batch).reshape(-1)
                    self.cached_batches.append(batch)
            self.cached_batches = torch.stack(self.cached_batches)
            np.random.seed(42)

    def __len__(self):
        return self.n_batch

    def __iter__(self):
        # for i_batch in range(min(self.n_batch, len(self.cached_batches))):
        # for i_batch in range(self.n_batch):
            if self.fix_seed:
                #print(len(self.cached_batches))
                for i_batch in range(self.n_batch):
                # for i_batch in range(min(self.n_batch, len(self.cached_batches))):
                    batch = self.cached_batches[i_batch]
                    yield batch
            else:
                for i_batch in range(self.n_batch):
                    batch = []
                    classes = np.random.choice(range(len(self.m_ind)), self.n_cls, False)
                    for c in classes:
                        l = self.m_ind[c]
                        if len(l) >= self.n_per:
                            pos = np.random.choice(range(len(l)), self.n_per, False)
                            batch.append(l[pos])
                        elif (len(l)>0 and len(l) < self.n_per):
                            pos = np.random.choice(range(len(l)), self.n_per, True)
                            batch.append(l[pos])
                       
                    batch = torch.stack(batch).reshape(-1)
                    yield batch