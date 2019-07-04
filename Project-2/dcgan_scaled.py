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
lr = 3e-04
beta1 = .5 
beta2 = .999
num_epochs = 50
img_size = 64
dropout = 0.5
labeled_ratio = 0.125
wd = 1e-04

device = 'cpu'
if torch.cuda.is_available():
	#move models to GPU 
	device = 'cuda'
print('Using {}'.format(device))

#the gelu activation function 
class gelu(nn.Module):
    def __init__(self):
        super(gelu, self).__init__()

    def forward(self, x):
        return gelu_fn(x)


def gelu_fn(x):
    return 0.5 * x * (1 + torch.tanh(0.7978845608028654*(x+0.044715*x**3)))

#plot the training losses for the GAN network
def plot_losses(losses):
	fig, ax = plt.subplots()
	losses = np.array(losses)
	plt.plot(losses.T[0], label='Discriminator', alpha=.5)
	plt.plot(losses.T[1], label='Generator', alpha=.5)
	plt.title('Training GAN Losses')
	plt.legend()
	plt.show()
	plt.savefig("figure1.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#plot the accuracy of the discriminator as classifier on the validation split 
def plot_acc(accs):
	plt.title('Discriminator test accuracy with labeled_ratio={}'.format(labeled_ratio))
	plt.xlabel('epochs')
	plt.ylabel('top-1 accuracy')
	plt.plot(accs, '-r')
	plt.show()
	plt.savefig("figure2.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

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
	plt.savefig("figure3.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

#helper function for viewing a list of passed in sample images
def view_samples(epoch, samples):
	fig, axes = plt.subplots(figsize=(16,4), nrows=2, ncols=8, sharey=True, sharex=True)
	for ax, img in zip(axes.flatten(), samples[epoch]):
		img = img.detach().cpu().numpy()
		img = np.transpose(img, (1, 2, 0))
		img = ((img +1)*255 / (2)).astype(np.uint8) # rescale to pixel range (0-255)
		ax.xaxis.set_visible(False)
		ax.yaxis.set_visible(False)
		im = ax.imshow(img.reshape((img_size,img_size,3)))
	plt.show()
	plt.savefig("figure4.png", dpi=None, facecolor='w', edgecolor='w',
        orientation='portrait', papertype=None, format=None,
        transparent=False, bbox_inches=None, pad_inches=0.1,
        frameon=None, metadata=None)

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

#Discriminator network 
class Discriminator(nn.Module):
	def __init__(self, conv_dim=32, num_classes=10):
		super(Discriminator, self).__init__()

		#complete init
		self.conv_dim = conv_dim 
		self.num_classes = num_classes

		self.features = nn.Sequential(
			nn.Dropout(dropout),

			#64x64 input 
			conv(3, int(conv_dim/2), 4, batch_norm=False), #first layer, no batch_norm
			gelu(),
			nn.Dropout(dropout),
			
			#32x32 out 
			conv(int(conv_dim/2), conv_dim, 4),
			gelu(),
			nn.Dropout(dropout),
			
			#16x16 out 
			conv(conv_dim, 2*conv_dim, 4),
			gelu(),
			nn.Dropout(dropout),

			#8x8 out 
			conv(2*conv_dim, 4*conv_dim, 4),
			gelu(),
			nn.Dropout(dropout),
			
			#4x4 out 
			conv(4*conv_dim, 8*conv_dim, 4),
			gelu(),
			nn.Dropout(0.1)
			)
	
		self.classifier = nn.Sequential(
			#to a denser layer
			nn.Dropout(dropout),
			nn.Linear(1024, conv_dim*4*4*4),
			gelu(),
			#another dense for fine parametarization
			nn.Dropout(dropout),
			nn.Linear(conv_dim*4*4*4, conv_dim*4*4*4),
			gelu(),
			#to 10 logits
			nn.Dropout(0.1),
			nn.Linear(conv_dim*4*4*4, num_classes),
			)
		self.fc = nn.Linear(1024, num_classes)


	def forward(self, x, matching=False):
		#all hidden layers with LeakyReLU activations 
		x = self.features(x)
	
		#flatten and pass to dense layer for classification
		f = x.view(-1, 1024)
		#print('Value of f:')
		#print(f.size())

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

		#first dense and bn layer
		self.fc = nn.Linear(z_size, conv_dim*(4**3))
		#self.bn = nn.BatchNorm2d(conv_dim*8)

		#transpose conv layers
		self.t_conv0 = deconv(8*conv_dim, 4*conv_dim, 4)
		self.t_conv1 = deconv(4*conv_dim, 2*conv_dim, 4)
		self.t_conv2 = deconv(2*conv_dim, conv_dim, 4)
		self.t_conv3 = deconv(conv_dim, 3, 4, batch_norm=False)

	def forward(self, x):
		#dense layer + reshape + bn
		#print(x.size())
		x = self.fc(x)
		#print(x.size())
		x = x.view(-1, self.conv_dim*8, 4, 4) #batch_size x 4 x 4 
		##x = gelu_fn(self.bn(x))
		#print(x.size())
		x = gelu_fn(self.t_conv0(x))
		#transpose conv layers with ReLUs
		x = gelu_fn(self.t_conv1(x))
		#print(x.size())
		x = gelu_fn(self.t_conv2(x))
		#print(x.size())
		x = torch.tanh(self.t_conv3(x))
		#print(x.size())
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
def batch_split(x, y, labeled_ratio):
	batch_size = x.size(0)
	cap = int(labeled_ratio * batch_size)
	return x[:cap], x[cap:], y[:cap]

def train(D, G, trainloader, testloader, d_opt, g_opt):
	samples, losses, accuracies = [], [], [] 
	print_step = 300 

	# Get some fixed data for sampling. These are images that are held
	# constant throughout training, and allow us to inspect the model's performance
	sample_size = 16
	fixed_z = np.random.uniform(-1, 1, size=(sample_size, z_size))
	fixed_z = torch.from_numpy(fixed_z).float().to(device)

	for epoch in range(num_epochs):
		epoch_D_loss, epoch_G_loss = 0., 0.
		for batch_idx, (imgs, labels) in enumerate(trainloader):
			batch_size = imgs.size(0)
			imgs = scale(imgs)
			#print(batch_size)
			#train the Discriminator
			d_opt.zero_grad()

			#step 1. Train with real Images
			#Compute Discriminator label-loss and unsup-real-loss
			imgs = imgs.to(device)
			labels = labels.to(device)

			##  FOR SEMI_SUPERVISED
			#print('The format of the images before is:')
			#print(imgs.size())
			#print(labels.size())
			
			label_imgs, unlabel_imgs, labels = batch_split(imgs, labels, labeled_ratio)
			
			#print('The format of the images after is:')
			#print(label_imgs.size())
			#print(unlabel_imgs.size())
			#print(labels.size())

			label_imgs = label_imgs.to(device)
			unlabel_imgs = unlabel_imgs.to(device)
			labels = labels.to(device)

			D_label_out = D(label_imgs)
			#print('After discriminator:')
			#print(D_label_out.size())
			#print(D_label_out)
			#print(labels.size())
			#print(labels)
			D_supervised_loss = F.cross_entropy(D_label_out, labels, reduction='sum')

			D_unlab_out = D(unlabel_imgs)
			D_real_loss = - torch.mean(LSE(D_unlab_out), 0) + torch.mean(F.softplus(LSE(D_unlab_out), 1), 0)
			
			'''
			D_out = D(imgs)
			D_supervised_loss = 0
			#D_real_loss = real_loss(D_out)
			D_real_loss = - torch.mean(LSE(D_out), 0) + torch.mean(F.softplus(LSE(D_out), 1), 0)
			'''
			#step 2. Train with fake Images 
			#Compute Discriminator unsup-fake-loss 
			z = np.random.uniform(-1, 1, size=(batch_size, z_size))
			z = torch.from_numpy(z).float().to(device)
			fake = G(z)

			#print('Generator now:')
			#print(fake.size())
			D_fake = D(fake.detach())
			#D_fake_loss = fake_loss(D_fake)
			D_fake_loss = torch.mean(F.softplus(LSE(D_fake), 1), 0)

			#addup all losses and go to backprop
			D_loss = D_supervised_loss + D_real_loss + D_fake_loss
			D_loss.backward()
			d_opt.step()

			epoch_D_loss += D_loss.item()

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

			epoch_G_loss += G_loss.item()
				
		epoch_D_loss /= batch_idx
		epoch_G_loss /= batch_idx
		#print losses 
		losses.append((epoch_D_loss, epoch_G_loss))
		print('Epoch [{:5d}/{:5d}] | D_loss: {:6.4f} | G_loss: {:6.4f}'.format(
				epoch+1, num_epochs, epoch_D_loss, epoch_G_loss))

		##AFTER EACH EPOCH
		#generate and save sample fake Images
		G.eval() #for generating samples 
		samples_z = G(fixed_z)
		samples.append(samples_z)
		G.train() #back to training mode 

		#test discriminator's accuracy on validation split
		test_acc = test(D, testloader)
		accuracies.append(test_acc)

		#save state dicts
		torch.save({
			'epoch' 		   		:	epoch, 
			'model_state_dict' 		:	D.state_dict(),
			'optimizer_state_dict'	:	d_opt.state_dict(),
			'loss'					:	D_loss
			}, 'discriminator00.pth')

		torch.save({
			'epoch'					:	epoch,
			'model_state_dict'		:	G.state_dict(),
			'optimizer_state_dict'	:	g_opt.state_dict(),
			'loss'					: 	G_loss
			}, 'generator00.pth')

	#Save and view some training generator samples
	with open('train_samples.pkl', 'wb') as f:
		pickle.dump(samples, f)
	view_samples(-1, samples)

	return losses, accuracies

def test(D, testloader):
	D.eval()
	correct, total = 0., 0.
	test_loss = 0
	for imgs, labels in testloader:
		imgs = imgs.to(device)
		labels = labels.to(device)

		out = D(imgs)
		test_loss += F.cross_entropy(out, labels, reduction='sum').item()
		predictions = out.data.max(1, keepdim=True)[1] 
		correct += predictions.eq(labels.data.view_as(predictions)).to(device).sum()
	test_loss /= len(testloader.dataset)
	accuracy = 100. * correct / len(testloader.dataset)
	print('Test set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
		test_loss, correct, len(testloader.dataset), accuracy))

	return accuracy

def main():
	#resize & turn to tensors
	tf = transforms.Compose([
					transforms.Resize(img_size),
					transforms.ToTensor()])

	#STL10 dataset 
	train_data = datasets.STL10(root='data/', split='train', download=True, transform=tf)
	test_data  = datasets.STL10(root='data/', split='test', download=True, transform=tf)
	trainloader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=False, 
								drop_last=True, num_workers=num_workers)
	testloader  = DataLoader(dataset=test_data, batch_size=batch_size, shuffle=False, 
								drop_last=True, num_workers=num_workers)


	#intatiate GAN
	D = Discriminator(conv_dim=conv_dim, num_classes=10).to(device)
	G = Generator(z_size=z_size, conv_dim=conv_dim).to(device)

	#define optimizers
	d_opt = torch.optim.Adam(D.parameters(), lr, [beta1,beta2], weight_decay=wd)
	g_opt = torch.optim.Adam(G.parameters(), lr, [beta1,beta2], weight_decay=wd)

	#train GAN
	losses, accuracies = train(D, G, trainloader, testloader, d_opt, g_opt)
	plot_losses(losses)
	plot_acc(accuracies)

if __name__ == '__main__':
	main()
