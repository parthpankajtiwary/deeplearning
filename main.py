import torch
import torch.nn as nn
import torch.optim
import time


import torchvision
import torchvision.transforms as transforms

import torchvision.models as models

def main():


	resnet18 = models.resnet18()
	alexnet = models.alexnet()
	vgg16 = models.vgg16()
	squeezenet = models.squeezenet1_0()
	densenet = models.densenet161()
	inception = models.inception_v3()
	#googlenet = models.googlenet(pretrained=True)

	# Choose the model
	choise = 1
	if choise == 0:
		model = alexnet
		#model = vgg16
		model.features = torch.nn.DataParallel(model.features)
		model.cuda()
	else:
		model = resnet18
		#model = squeezenet
		#model = densenet
		#model = inception
		model = torch.nn.DataParallel(model).cuda()



	normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
									 std=[0.229, 0.224, 0.225])

	transform = transforms.Compose([transforms.ToTensor(), normalize])

	trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
	trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

	#testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
	#testloader = torch.utils.data.DataLoader(testset, batch_size=4, shuffle=False, num_workers=2)

	classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	print(device)
	# Set the learning rate and weight decay
	lr = 0.1
	wd = 1e-4
	momentum = 0.9

	criterion = nn.CrossEntropyLoss().cuda()
	optimizer = torch.optim.SGD(model.parameters(), lr,
								momentum=momentum,
								weight_decay=wd)
	# Train for 20 epochs
	for j in range(1,20):
		train(model, trainloader, criterion, optimizer)
	#test(model, testloader, criterion, optimizer)

def train(model, trainloader, criterion, optimizer):
	batch_time = AverageMeter()
	data_time = AverageMeter()
	losses = AverageMeter()
	top1 = AverageMeter()
	top5 = AverageMeter()

	# switch to train mode
	model.train()

	end = time.time()
	for i, (input, target) in enumerate(trainloader):
		# measure data loading time
		data_time.update(time.time() - end)

		target = target.cuda(async=True)
		input_var = torch.autograd.Variable(input)
		target_var = torch.autograd.Variable(target)

		# compute output
		output = model(input_var)
		loss = criterion(output, target_var)

		# measure accuracy and record loss
		prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
		losses.update(loss.data[0], input.size(0))
		top1.update(prec1[0], input.size(0))
		top5.update(prec5[0], input.size(0))

		# compute gradient and do SGD step
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()

		# measure elapsed time
		batch_time.update(time.time() - end)
		end = time.time()

		if i % 200 == 1:
			print('Epoch: [{0}][{1}/{2}]\t'
				  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
				  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
				  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
				  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
				  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
				   epoch, i, len(train_loader), batch_time=batch_time,
				   data_time=data_time, loss=losses, top1=top1, top5=top5))

	print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'.format(top1=top1, top5=top5))

def test(model, testloader, criterion, optimizer):
	pass

class AverageMeter(object):
	"""Computes and stores the average and current value"""
	def __init__(self):
		self.reset()

	def reset(self):
		self.val = 0
		self.avg = 0
		self.sum = 0
		self.count = 0

	def update(self, val, n=1):
		self.val = val
		self.sum += val * n
		self.count += n
		self.avg = self.sum / self.count

def adjust_learning_rate(optimizer, epoch):
	"""Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
	lr = args.lr * (0.1 ** (epoch // 30))
	for param_group in optimizer.param_groups:
		param_group['lr'] = lr


def accuracy(output, target, topk=(1,)):
	"""Computes the precision@k for the specified values of k"""
	maxk = max(topk)
	batch_size = target.size(0)

	_, pred = output.topk(maxk, 1, True, True)
	pred = pred.t()
	correct = pred.eq(target.view(1, -1).expand_as(pred))

	res = []
	for k in topk:
		correct_k = correct[:k].view(-1).float().sum(0)
		res.append(correct_k.mul_(100.0 / batch_size))
	return res
if __name__ == '__main__':
	main()

