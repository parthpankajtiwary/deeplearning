from matplotlib import pyplot as plt 
import numpy as np 
import pickle 

import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import DataLoader
from torchvision import datasets, transforms 

#hyperparams 
batch_size = 128
num_workers = 0 
conv_dim = 32 
z_size = 100
lr = 1e-04
beta1 = .5 
beta2 = .999
num_epochs = 100
img_size = 32 
dropout = 0.

device = 'cpu'
if torch.cuda.is_available():
	#move models to GPU 
	device = 'cuda'
print('Using {}'.format(device))

#plot the training losses for the GAN network
def plot_losses(losses):
	fig, ax = plt.subplots()
	losses = np.array(losses)
	plt.plot(losses.T[0], label='Discriminator', alpha=.5)
	plt.plot(losses.T[1], label='Generator', alpha=.5)
	plt.title('Training GAN Losses')
	plt.legend()
	plt.show()

#plot some sample images with their labels
def plot_imgs(dataloader):
	dataiter = iter(dataloader)
	imgs, labels = dataiter.next() 

	fig = plt.figure(figsize=(25,4)) 
	plot_size = 20 
	for idx in np.arange(plot_size):
		ax = fig.add_subplot(2, plot_size/2, idx+1, xticks=[], yticks=[])
		ax.imshow(np.transpose(imgs[idx], (1,2,0)))
		#print the label also 
		ax.set_title(str(labels[idx].item()))
	plt.show()

#helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
	fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
	for ax, img in zip(axes.flatten(), samples[epoch]):
		img = img.detach().cpu().numpy()
		img = np.transpose(img, (1, 2, 0))
		img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		im = ax.imshow(img.reshape((32,32,3)))
	plt.show()

#scale an image to floats in (-1,1)
def scale(x, feat_range=(-1,1)):
	''' Scale takes in an image x and returns that image, scaled
       with a feature_range of pixel values from -1 to 1. 
       This function assumes that the input x is already scaled from 0-1.'''
    # assume x is scaled to (0, 1)
    # scale to feature_range and return scaled x
	min, max = feat_range
	return x * (max - min) + min

#convolution function 
def conv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
	"""Creates a convolutional layer, with optional batch normalization.
	"""
	layers = []
	conv_layer = nn.Conv2d(in_channels, out_channels, 
							kernel_size, stride, padding, bias=False)

	# append conv layer
	layers.append(conv_layer)

	if batch_norm:
		# append batchnorm layer
		layers.append(nn.BatchNorm2d(out_channels))
 
	# using Sequential container
	return nn.Sequential(*layers)

#deconvolution function 
def deconv(in_channels, out_channels, kernel_size, stride=2, padding=1, batch_norm=True):
	"""Creates a transposed-convolutional layer, with optional batch normalization.
    """
    # create a sequence of transpose + optional batch norm layers
	layers = []
	transpose_conv_layer = nn.ConvTranspose2d(in_channels, out_channels, 
	                                          kernel_size, stride, padding, bias=False)
	# append transpose convolutional layer
	layers.append(transpose_conv_layer)

	if batch_norm:
		# append batchnorm layer
		layers.append(nn.BatchNorm2d(out_channels))

	return nn.Sequential(*layers)

def out_from_logits(x):
	# new discriminator output based on 
	# the "Improving Techniques for GANs" paper
	x = torch.sum(torch.exp(x), dim=1)
	x = x / (x+1)
	return x

def real_loss(D_out, smooth=False):
	batch_size_ = D_out.size(0)
	# label smoothing
	if smooth:
		# smooth, real labels = 0.9
		labels = torch.ones(batch_size_)*0.9
	else:
		labels = torch.ones(batch_size_) # real labels = 1

	#move labels to GPU if available     
	labels = labels.to(device)

	#binary cross entropy with logits loss
	criterion = nn.BCEWithLogitsLoss().to(device)

	#calculate loss
	loss = criterion(out_from_logits(D_out), labels)

	return loss

def fake_loss(D_out):
	batch_size_ = D_out.size(0)
	labels = torch.zeros(batch_size_).to(device) # fake labels = 0

	criterion = nn.BCEWithLogitsLoss().to(device)

	# calculate loss
	loss = criterion(out_from_logits(D_out), labels)

	return loss

def supervised_loss(D_out, labels, mask):
	batch_size = D_out.size(0)

	#cross-entropy loss for regular supervised multi-classification loss
	criterion = nn.CrossEntropyLoss().to(device)

	#apply labeling mask to hide unwanted labels
	NotImplemented
	loss = criterion(mask(D_out.squeeze()), labels)

	return loss 

#Discriminator network 
class Discriminator(nn.Module):
	def __init__(self, conv_dim=32, num_classes=10):
		super(Discriminator, self).__init__()

		#complete init
		self.conv_dim = conv_dim 
		self.num_classes = num_classes

		self.features = nn.Sequential(
			#nn.Dropout(dropout),
			#32x32 input 
			conv(3, conv_dim, 4, batch_norm=False), #first layer, no batch_norm
			nn.LeakyReLU(0.2, inplace=True), 
			nn.Dropout(dropout),
			#16x16 out 
			conv(conv_dim, 2*conv_dim, 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(dropout),
			#8x8 out 
			conv(2*conv_dim, 4*conv_dim, 4),
			nn.LeakyReLU(0.2, inplace=True),
			nn.Dropout(dropout),
			#4x4 out 
			conv(4*conv_dim, 8*conv_dim, 4, batch_norm=False),
			#nn.LeakyReLU(0.2, inplace=True),
			#nn.Dropout(dropout),			
			#2x2 out
			)

		self.fc = nn.Linear(conv_dim*(4**4), num_classes)


	def forward(self, x, matching=False):
		#all hidden layers with LeakyReLU activations 
		x = self.features(x)

		#flatten 
		f = x.view(-1, self.conv_dim*(4**4))
	
		x = self.fc(f)

		if matching:
			return f, x
		else:
			return x 

#Generator network
class Generator(nn.Module):
	def __init__(self, z_size, conv_dim=32):
		super(Generator, self).__init__()
		self.conv_dim = conv_dim

		#first dense layer
		self.fc = nn.Linear(z_size, conv_dim*(4**3))

		#transpose conv layers 
		self.t_conv1 = deconv(4*conv_dim, conv_dim*2, 4)
		self.t_conv2 = deconv(2*conv_dim, conv_dim, 4)
		self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

	def forward(self, x):
		#dense layer + reshape 
		x = self.fc(x)
		x = x.view(-1, self.conv_dim*4, 4, 4) #batch_size x 4 x 4 

		#transpose conv layers with ReLUs
		x = F.relu(self.t_conv1(x))
		x = F.relu(self.t_conv2(x))
		x = torch.tanh(self.t_conv3(x))

		return x

def to_scalar(var):
	# returns a python float
	return var.view(-1).data.tolist()[0]

def argmax(vec):
	# return the argmax as a python int
	_, idx = torch.max(vec, 1)
	return to_scalar(idx)

#log_sum_exp function
def LSE(before_softmax_output):
	# exp = torch.exp(before_softmax_output)
	# sum_exp = torch.sum(exp,1) #right
	# log_sum_exp = torch.log(sum_exp)
	# return log_sum_exp
	vec = before_softmax_output
	max_score = vec[0, argmax(vec)]
	max_score_broadcast = max_score.view(1, -1).expand(1, vec.size()[1])
	output = max_score + torch.log(torch.sum(torch.exp(vec - max_score_broadcast),1))
	return output

#split every batch to labeled+unlabeled batch for semi-supervised-mode
def batch_split(x, y, labeled_ratio = .25):
	batch_size = x.size(0)
	return x[:labeled_ratio * batch_size], x[labeled_ratio * batch_size:], y[:labeled_ratio * batch_size]

def train(D, G, trainloader, testloader, d_opt, g_opt):
	samples, losses = [], [] 
	print_step = 300 

	# Get some fixed data for sampling. These are images that are held
	# constant throughout training, and allow us to inspect the model's performance
	sample_size = 16
	fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
	fixed_z = torch.from_numpy(fixed_z).float().to(device)

	for epoch in range(num_epochs):
		for batch_idx, (imgs, labels) in enumerate(trainloader):
			batch_size = imgs.size(0)
			imgs = scale(imgs)

			#train the Discriminator
			d_opt.zero_grad()

			#step 1. Train with real Images
			#Compute Discriminator label-loss and unsup-real-loss
			imgs = imgs.to(device)
			labels = labels.to(device)

			'''  FOR SEMI_SUPERVISED

			label_imgs, unlabel_imgs, labels = batch_split(imgs, labels)

			D_label_out = D(label_imgs)
			D_supervised_loss = nn.CrossEntropyLoss(D_label_out, labels)

			D_unlab_out = D(unlabel_imgs)
			D_real_loss = - torch.mean(LSE(D_unlab_out), 0) + torch.mean(F.softplus(LSE(D_unlab_out), 1), 0)
			'''

			D_out = D(imgs)
			D_supervised_loss = 0
			#D_real_loss = real_loss(D_out)
			D_real_loss = - torch.mean(LSE(D_out), 0) + torch.mean(F.softplus(LSE(D_out), 1), 0)

			#step 2. Train with fake Images 
			#Compute Discriminator unsup-fake-loss 
			z = np.random.uniform(-1, 1, size=(batch_size, z_size))
			z = torch.from_numpy(z).float().to(device)
			fake = G(z)
			D_fake = D(fake.detach())
			#D_fake_loss = fake_loss(D_fake)
			D_fake_loss = torch.mean(F.softplus(LSE(D_fake), 1), 0)

			#addup all losses and go to backprop
			D_loss = D_supervised_loss + D_real_loss + D_fake_loss
			D_loss.backward()
			d_opt.step()

			#train the Generator
			g_opt.zero_grad()

			#step 1. Train with fake Images and flipped labels 
			#Generate fake Images 
			z = np.random.uniform(-1, 1, size=(batch_size, z_size))
			z = torch.from_numpy(z).float().to(device)

			# Compute the Generator losses on fake images 
			# using feature matching
			fake = G(z)
			#D_fake = D(fake)
			#G_loss = real_loss(D_fake) #use real loss to flip labels 
			D_real_feats, D_real_out = D(imgs.detach(), matching=True)
			D_fake_feats, D_fake_out = D(fake, matching=True)
			D_real_feats = torch.mean(D_real_feats, 0)
			D_fake_feats = torch.mean(D_fake_feats, 0)
			G_loss = torch.mean(torch.abs(D_real_feats.detach() - D_fake_feats))
			
			#backprop
			G_loss.backward()
			g_opt.step()

			#Print some Loss stuff 
			if batch_idx % print_step == 0:
				#append Discriminator losses and Generator losses
				losses.append((D_loss.item(), G_loss.item()))
				
				#print losses 
				print('Epoch [{:5d}/{:5d}] | D_loss: {:6.4f} | G_loss: {:6.4f}'.format(
					epoch+1, num_epochs, D_loss.item(), G_loss.item()))


		##AFTER EACH EPOCH
		#generate and save sample fake Images
		G.eval() #for generating samples 
		samples_z = G(fixed_z)
		samples.append(samples_z)
		G.train() #back to training mode 

		#test discriminator's accuracy on test splti
		#test_acc = test(D, testloader)

		#save state dicts
		torch.save({
			'epoch' 		   		:	epoch, 
			'model_state_dict' 		:	D.state_dict(),
			'optimizer_state_dict'	:	d_opt.state_dict(),
			'loss'					:	D_loss
			}, '/discriminator.pth')

		torch.save({
			'epoch'					:	epoch,
			'model_state_dict'		:	G.state_dict(),
			'optimizer_state_dict'	:	g_opt.state_dict(),
			'loss'					: 	G_loss
			}, '/generator.pth')

	#Save and view some training generator samples
	with open('train_samples.pkl', 'wb') as f:
		pickle.dump(samples, f)
	view_samples(-1, samples)

	return losses

def test(D, testloader):
	D.eval()
	correct, total = 0., 0.
	test_loss = 0
	for imgs, labels in testloader:
		imgs = imgs.to(device)
		labels = labels.to(device)

		out = D(imgs)
		test_loss += F.cross_entropy(out, labels, size_average=False).item()
		predictions = D.data.max(1, keepdim=True)[1] 
		correct += predictions.eq(labels.data.view_as(predictions)).to(device).sum()
	test_loss /= len(testloader.dataset)
	accuracy = 100. * correct / len(testloader.dataset)
	print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(testloader.dataset), accuracy))

	return accuracy

def main():
	#resize & turn to tensors
	tf = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor()])

	#SVHN dataset 
	train_data = datasets.SVHN(root='data/', split='train', download=True, transform=tf)
	test_data  = datasets.SVHN(root='data/', split='test', download=True, transform=tf)
	trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, 
								drop_last=True, num_workers=num_workers)
	testloader  = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, 
								drop_last=True, num_workers=num_workers)

	#intatiate GAN
	D = Discriminator(conv_dim=conv_dim, num_classes=10).to(device)
	G = Generator(z_size=z_size, conv_dim=conv_dim).to(device)

	#define optimizers
	d_opt = torch.optim.Adam(D.parameters(), lr, [beta1,beta2])
	g_opt = torch.optim.Adam(G.parameters(), lr, [beta1,beta2])

	#train GAN
	losses = train(D, G, trainloader, testloader, d_opt, g_opt)
	plot_losses(losses)

if __name__ == '__main__':
	main()
