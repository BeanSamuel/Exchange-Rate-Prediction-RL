import torch


class Dataset:
    def __init__(self, batch_size, minibatch_size, device):
        self.batch_size = batch_size
        self.minibatch_size = minibatch_size
        self.device = device
        # self.size = self.batch_size // self.minibatch_size
        self._idx_buf = torch.randperm(batch_size)

    def update(self, datas):
        self.datas = datas

    def __len__(self):
        return self.batch_size // self.minibatch_size

    def __getitem__(self, idx):
        start = idx * self.minibatch_size
        end = (idx + 1) * self.minibatch_size
        sample_idx = self._idx_buf[start:end]

        data_dict = {}
        for k, v in self.datas.items():
            if v is not None:
                data_dict[k] = v[sample_idx].detach()
                
        if end >= self.batch_size:
            self._shuffle_idx_buf()

        return data_dict

    def _shuffle_idx_buf(self):
        self._idx_buf[:] = torch.randperm(self.batch_size)
