import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, TensorDataset

from utils_sub import clip_grad

import numpy as np
import copy

class FedAvgClient(nn.Module):
	"""
	1. Receive global model from server
	2. Perform local training (get gradients)
	3. Return local model to server
	"""

	def __init__(self, model, output_size, data, lr, epoch, batch_size, clip, sigma,important_feature, device = None):
		"""
		model:   ML model
		data:    (tuple) dataset, all data in client side is used as training data
		lr:      learning rate
		epoch:   epoch of local update
		"""
		super(FedAvgClient, self).__init__()

		self.device = device
		self.batch_size = batch_size
		torch_dataset = TensorDataset(torch.tensor(data[0]), torch.tensor(data[1]))
		self.data_size = len(torch_dataset)
		self.data_loader = DataLoader(
			dataset = torch_dataset,
			batch_size = self.batch_size,
			shuffle = True)
		self.lr = lr
		self.epoch = epoch
		self.clip = clip
		self.sigma = sigma
		self.model = model(data[0].shape[1], output_size).to(self.device)
		self.recv_model = model(data[0].shape[1], output_size).to(self.device)


	def recv(self, model_para):
		"""
		收到合并模型
		"""
		self.model.load_state_dict(copy.deepcopy(model_para))
		self.recv_model.load_state_dict(copy.deepcopy(model_para))

	def update(self):
		"""
		local model update
		"""
		self.model.train()
		"""
		model.train()的作用是启用 Batch Normalization 和 Dropout。

		如果模型中有BN层(Batch Normalization）和Dropout，需要在训练时添加model.train()。
		model.train()是保证BN层能够用到每一批数据的均值和方差。对于Dropout，model.train()
		是随机取一部分网络连接来训练更新参数。
		"""
		criterion = nn.CrossEntropyLoss()#交叉熵损失函数CrossEntropyLoss()
		optimizer = torch.optim.SGD(self.model.parameters(), lr = self.lr, momentum = 0.9)
		"""
		SGD 就是随机梯度下降

		Momentum 
		传统的参数 W 的更新是把原始的 W 累加上一个负的学习率(learning rate) 乘以校正值 (dx). 此方法比较曲折。
		我们把这个人从平地上放到了一个斜坡上, 只要他往下坡的方向走一点点, 由于向下的惯性, 他不自觉地就一直往下走, 走的弯路也变少了. 这就是 Momentum 参数更新。


		"""
		for e in range(self.epoch):#每次更新三个唯一一个单位
			for batch_x, batch_y in self.data_loader:
				batch_x, batch_y = batch_x.to(self.device), batch_y.to(self.device)
				pred_y = self.model(batch_x.float())
				loss = criterion(pred_y, batch_y.long()) / len(self.data_loader)
				loss.backward()
			# bound 12 sensitivity (gradient clippling)
			grads = dict(self.model.named_parameters())
			for name in grads:

				grads[name].grad = clip_grad(grads[name].grad, self.clip)

			optimizer.step()
			optimizer.zero_grad()
			"""
			这三个函数的作用是先将梯度归零（optimizer.zero_grad()），然后反向传播计算得到每个参数的梯度值（loss.backward()），最后通过梯度下降执行一步参数更新（optimizer.step()）
			"""
		new_param = copy.deepcopy(self.model.state_dict())
		self.model.load_state_dict(copy.deepcopy(new_param))

class FedAvgServer(nn.Module):
	""" Server of Federated Learning
	1. Receive model (or gradients) from clients
	2. Aggregate local models (or gradients)
	3. Compute global model, broadcast global model to clients
	"""
	def __init__(self, fl_par):
		super(FedAvgServer, self).__init__()

		self.device = fl_par['device']
		self.client_num = fl_par['client_num']
		self.C = fl_par['C']  # (float) C in [0, 1]
		self.clip = fl_par['clip']

		self.data = []
		self.target = []
		#下面是吧test数据集取出来
		for sample in fl_par['data'][self.client_num:]:
			self.data += [torch.tensor(sample[0]).to(self.device)]  # test set
			self.target += [torch.tensor(sample[1]).to(self.device)]  # target label


		self.input_size = int(self.data[0].shape[1])#[tensor]
		self.lr = fl_par['lr']

		self.clients = [FedAvgClient(fl_par['model'],
									 fl_par['output_size'],
									 fl_par['data'][i],
									 fl_par['lr'],
									 fl_par['epoch'],
									 fl_par['batch_size'],
									 fl_par['clip'],
									 fl_par['sigma'],
									 self.device)
							for i in range(self.client_num)]
		
		self.global_model = fl_par['model'](self.input_size, fl_par['output_size']).to(self.device)
		print(self.global_model)
		self.weight = np.array([client.data_size * 1.0 for client in self.clients])#[6512. 6512. 6512. 6512. 6512.]
		self.broadcast(self.global_model.state_dict())

	def aggregated(self, idxs_users):
		"""
		FedAvg
		"""
		model_par = [self.clients[idx].model.state_dict() for idx in idxs_users]
		new_par = copy.deepcopy(model_par[0])
		for name in new_par:
			new_par[name] = torch.zeros(new_par[name].shape).to(self.device)
		for idx, par in enumerate(model_par):
			w = self.weight[idxs_users[idx]] / np.sum(self.weight[:])
			for name in new_par:
				 # new_par[name] += par[name] * (self.weight[idxs_users[idx]] / np.sum(self.weight[idxs_users]))
				new_par[name] += par[name] * (w / self.C)
		self.global_model.load_state_dict(copy.deepcopy(new_par))
		return self.global_model.state_dict().copy()

	def broadcast(self, new_par):
		"""
		Send aggregated model to all clients
		"""
		for client in self.clients:
			client.recv(new_par.copy())

	def test_acc(self):
		"""
		compute accuracy using test set
		"""
		self.global_model.eval()
		t_pred_y = self.global_model(self.data.float())
		_, predicted = torch.max(t_pred_y, 1)

		acc = (predicted == self.target).sum().item() / self.target.size(0)
		return acc

	def test_acc_global(self):
		self.global_model.eval()
		correct = 0
		tot_sample = 0
		for i in range(len(self.data)):
			t_pred_y = self.global_model(self.data[i].float())
			_, predicted = torch.max(t_pred_y, 1)
			correct += (predicted == self.target[i]).sum().item()
			tot_sample += self.target[i].size(0)
		acc = correct / tot_sample
		return acc

	def global_update(self):
		idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
		for idx in idxs_users:
			self.clients[idx].update()
		self.broadcast(self.aggregated(idxs_users))
		# acc = self.test_acc()
		acc = self.test_acc_global()
		return acc


	def aggregated_grad(self, idxs_users, grads):
		"""
		FedAvg - Update model using gradients
		"""
		agg_grad = copy.deepcopy(grads[0])
		for name in agg_grad:
			agg_grad[name] = torch.zeros(agg_grad[name].shape).to(self.device)
		for name in self.global_model.state_dict():
			self.global_model.state_dict()[name] -= agg_grad[name]
		return self.global_model.state_dict().copy()

	def global_update_grad(self):
		for e in range(self.epoch):
			idxs_users = np.random.choice(range(len(self.clients)), int(self.C * len(self.clients)), replace=False)
			grads = []
			for idx in idxs_users:
				grads.append(copy.deepcopy(self.clients[idx].update_grad()))
			self.broadcast(self.aggregated_grad(idxs_users, grads))
			acc = self.test_acc()
			print("global epochs = {:d}, acc = {:.4f}".format(e + 1, acc))

	def set_lr(self,lr):
		for c in self.clients:
			c.lr =lr
	
		