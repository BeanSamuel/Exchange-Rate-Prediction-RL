import torch
import wandb


# class WandbLogger:
#     def __init__(self, project, run_name, log=True):
#         self.data = {}
#         self.data_cnt = {}
#         self.is_log = log
#         if log:
#             wandb.init(project=project)
#             wandb.run.name = run_name
#             wandb.run.save()

    # def stop(self):
    #     wandb.finish()

#     def log(self, datas: dict):
#         if self.is_log:
#             for k, v in datas.items():
#                 if isinstance(v, torch.Tensor):
#                     if v.nelement == 0:
#                         v = torch.nan
#                     v = v.mean().item()
#                 n = self.data_cnt.get(k, 0)
#                 x = self.data.get(k, 0)
#                 self.data_cnt[k] = n + 1
#                 self.data[k] = x * n / (n+1) + v / (n+1)

#     def upload(self):
#         if self.is_log:
#             wandb.log(self.data)
#         self.data.clear()
#         self.data_cnt.clear()
class WandbLogger:
    def __init__(self, project, run_name, log=True):
        pass

    def log(self, datas: dict):
        pass

    def upload(self):
        pass
    def stop(self):
        pass