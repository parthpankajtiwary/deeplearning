from matplotlib import pyplot as plt
import numpy as np

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, sampler
from torchvision import datasets, transforms

from improving_dcgan import *

#hyperparams
batch_size = 128
num_workers = 0
conv_dim = 32
labels_ratio = .125
lr = 1e-04
beta1 = .5
beta2 = .999
num_epochs = 25
img_size = 32
dropout = 0.5
printstep = 50
wd = 1e-04

device = 'cpu'
if torch.cuda.is_available():
	#move models to GPU
	device = 'cuda'
print('Using {}'.format(device))


def train(D, trainloader, testloader, opt, epoch0):
	train_losses, test_losses, accuracies = [], [], []
	for epoch in range(num_epochs):
		D.train()
		epoch_loss = 0.
		for batch_idx, (imgs, labels) in enumerate(trainloader):
			imgs = scale(imgs).to(device)
			labels = labels.to(device)

			opt.zero_grad()
			out = D(imgs)
			loss = F.cross_entropy(out, labels, reduction='sum').to(device)
			loss.backward()
			opt.step()

			epoch_loss += loss.item()
		epoch_loss /= batch_idx
		train_losses.append(epoch_loss)

		D.eval()
		correct, test_loss = 0., 0.
		for batch_idx, (imgs, labels) in enumerate(testloader):
			imgs = scale(imgs).to(device)
			labels = labels.to(device)

			out = D(imgs)
			loss = F.cross_entropy(out, labels, reduction='sum').to(device)
			predictions = out.data.max(1, keepdim=True)[1]
			correct += predictions.eq(labels.data.view_as(predictions)).to(device).sum()

			test_loss += loss.item()
		test_loss /= batch_idx
		accuracy = 100. * correct / len(testloader.dataset)
		print('Epoch [{:5d}/{:5d}] | train_loss: {:6.4f} | test_loss: {:6.4f} | test_accuracy: {:6.4f}'.format(
					epoch0+epoch+1, epoch0+num_epochs, epoch_loss, test_loss, accuracy))
		accuracies.append(accuracy)
		test_losses.append(test_loss)

	return train_losses, test_losses, accuracies

def train_discriminator(labeled_ratio, only_dense=False):
	#resize & turn to tensors
	tf = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor()])

	#SVHN dataset
	train_data = datasets.SVHN(root='data/', split='train', download=True, transform=tf)
	test_data  = datasets.SVHN(root='data/', split='test', download=True, transform=tf)

	#define number of labels used
	indices = np.arange(labeled_ratio * len(train_data), dtype=np.int32)

	trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False,
								sampler=sampler.SubsetRandomSampler(indices),
								drop_last=True, num_workers=num_workers)
	testloader  = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False,
								drop_last=True, num_workers=num_workers)

	#load Discriminator network
	# checkpoint_d = torch.load('checkpoints/discriminator50.pth')
	D = Discriminator(conv_dim).to(device)
	d_opt = torch.optim.Adam(D.parameters(), lr, [beta1,beta2], weight_decay=wd)
	# D.load_state_dict(checkpoint_d['model_state_dict'])
	# d_opt.load_state_dict(checkpoint_d['optimizer_state_dict'])
	epoch0 = 0
	# loss = checkpoint_d['loss']

	if only_dense:
		for param in D.features.parameters():
			param.requires_grad = False

	train_losses, test_losses, accs = train(D, trainloader, testloader, d_opt, epoch0)

	plt.title('Discriminator supervised training (feats+dense)')
	plt.xlabel('epochs')
	plt.ylabel('cross-entropy loss')
	plt.plot(list(np.arange(epoch0, epoch0+num_epochs)), train_losses, '-b', label='train')
	plt.plot(list(np.arange(epoch0, epoch0+num_epochs)), test_losses, '-r', label='test')
	plt.legend()
	plt.show()

	return accs

if __name__ == '__main__':
	train_discriminator(labels_ratio)
