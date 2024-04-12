from __future__ import print_function
import matplotlib.pyplot as plt
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, OneCycleLR
import albumentations as A
from albumentations.pytorch import ToTensorV2

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'
import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)


def plot_performance(train_losses, train_acc, test_losses, test_acc):
  fig, axs = plt.subplots(2,2,figsize=(15,10))
  axs[0, 0].plot(train_losses)
  axs[0, 0].set_title("Training Loss")
  axs[1, 0].plot(train_acc)
  axs[1, 0].set_title("Training Accuracy")
  axs[0, 1].plot(test_losses)
  axs[0, 1].set_title("Test Loss")
  axs[1, 1].plot(test_acc)
  axs[1, 1].set_title("Test Accuracy")


def plot_data(loader):
  batch_data, batch_label = next(iter(loader))
  fig = plt.figure()
  classes = ('plane', 'car', 'bird', 'cat',
             'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
  for i in range(12):
      plt.subplot(3,4,i+1)
      plt.tight_layout()
      plt.imshow(np.transpose(batch_data[i].numpy(), (2, 1, 0)), cmap='gray')
      plt.title(f'{classes[batch_label[i].item()]}')
      plt.xticks([])
      plt.yticks([])
  plt.show()


def get_misclassified_predictions(model, test_loader, device):
  model.eval()
  test_loss = 0
  correct = 0
  i =0
  misclassified_dict = []
  with torch.no_grad():
      for data, target in test_loader:
          data, target = data.to(device), target.to(device)
          output = model(data)
          #test_loss += nn.CrossEntropyLoss(output, target, reduction='sum').item()  # sum up batch loss
          pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
          t = pred.eq(target.view_as(pred)).view(-1, target.size(0))
          t = t.cpu().numpy().reshape(-1)
          for ind,val in enumerate(t):
            if not val :
              temp_dict = {}
              temp_dict['data'] = data[ind].cpu().numpy()
              temp_dict['target'] = target[ind]
              temp_dict['pred'] = pred[ind]
              misclassified_dict.append(temp_dict)
  return misclassified_dict


def plot_misclassified_predictions(misclassified_dict, n=10):
  classes = ['plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']
  display_images = misclassified_dict[:n]
  index = 0
  fig = plt.figure(figsize=(10,5))
  for img in display_images:
      image = img['data']
      pred = classes[img['pred']]
      actual = classes[img['target']]
      ax = fig.add_subplot(2, 5, index+1)
      ax.axis('off')
      ax.set_title(f'\n Predicted Label : {pred} \n Target Label : {actual}',fontsize=10)
      ax.imshow(np.transpose(image, (1, 2, 0)))
      index = index + 1
  plt.show()


class GradCAM:
  """ Class for extracting activations and 
  registering gradients from targetted intermediate layers 
  target_layers = list of convolution layer index as shown in summary
  """
  def __init__(self, model, candidate_layers=None):
      def save_fmaps(key):
        def forward_hook(module, input, output):
            self.fmap_pool[key] = output.detach()
        return forward_hook
      def save_grads(key):
        def backward_hook(module, grad_in, grad_out):
            self.grad_pool[key] = grad_out[0].detach()
        return backward_hook
      self.device = next(model.parameters()).device
      self.model = model
      self.handlers = []  # a set of hook function handlers
      self.fmap_pool = {}
      self.grad_pool = {}
      self.candidate_layers = candidate_layers  # list
      for name, module in self.model.named_modules():
          if self.candidate_layers is None or name in self.candidate_layers:
              self.handlers.append(module.register_forward_hook(save_fmaps(name)))
              self.handlers.append(module.register_backward_hook(save_grads(name)))

  def _encode_one_hot(self, ids):
      one_hot = torch.zeros_like(self.nll).to(self.device)
      print(one_hot.shape)
      one_hot.scatter_(1, ids, 1.0)
      return one_hot

  def forward(self, image):
      self.image_shape = image.shape[2:] # HxW
      self.nll = self.model(image)
      #self.probs = F.softmax(self.logits, dim=1)
      return self.nll.sort(dim=1, descending=True)  # ordered results

  def backward(self, ids):
      """
      Class-specific backpropagation
      """
      one_hot = self._encode_one_hot(ids)
      self.model.zero_grad()
      self.nll.backward(gradient=one_hot, retain_graph=True)

  def remove_hook(self):
      """
      Remove all the forward/backward hook functions
      """
      for handle in self.handlers:
          handle.remove()

  def _find(self, pool, target_layer):
      if target_layer in pool.keys():
          return pool[target_layer]
      else:
          raise ValueError("Invalid layer name: {}".format(target_layer))

  def generate(self, target_layer):
      fmaps = self._find(self.fmap_pool, target_layer)
      grads = self._find(self.grad_pool, target_layer)
      weights = F.adaptive_avg_pool2d(grads, 1)
      gcam = torch.mul(fmaps, weights).sum(dim=1, keepdim=True)
      gcam = F.relu(gcam)
      # need to capture image size duign forward pass
      gcam = F.interpolate(
          gcam, self.image_shape, mode="bilinear", align_corners=False
      )
      # scale output between 0,1
      B, C, H, W = gcam.shape
      gcam = gcam.view(B, -1)
      gcam -= gcam.min(dim=1, keepdim=True)[0]
      gcam /= gcam.max(dim=1, keepdim=True)[0]
      gcam = gcam.view(B, C, H, W)
      return gcam

def generate_gradcam(misclassified_images, model, target_layers,device):
    images=[]
    labels=[]
    for i, dct in enumerate(misclassified_images):
      img = torch.tensor(dct['data'])
      correct = dct['target']
      images.append(img)
      labels.append(correct)

    model.eval()
    
    # map input to device
    images = torch.stack(images).to(device)
    
    # set up grad cam
    gcam = GradCAM(model, target_layers)
    
    # forward pass
    probs, ids = gcam.forward(images)
    
    # outputs agaist which to compute gradients
    ids_ = torch.LongTensor(labels).view(len(images),-1).to(device)
    
    # backward pass
    gcam.backward(ids=ids_)
    layers = []
    for i in range(len(target_layers)):
        target_layer = target_layers[i]
        print("Generating Grad-CAM @{}".format(target_layer))
        # Grad-CAM
        layers.append(gcam.generate(target_layer=target_layer))
        
    # remove hooks when done
    gcam.remove_hook()
    return layers, probs, ids

def load_data():
    transform = transforms.Compose(
      [transforms.ToTensor()])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                          download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=4,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=4,
                                          shuffle=False, num_workers=2)
    return trainloader, trainset    
  
def plot_gradcam(gcam_layers, target_layers, class_names, image_size,predicted, misclassified_images):
    trl, trs = load_data()
    
    mean = list(np.round(trs.data.mean(axis=(0,1,2))/255, 4))
    std = list(np.round(trs.data.std(axis=(0,1,2))/255,4))

    images=[]
    labels=[]
    for i, dct in enumerate(misclassified_images):
      img = torch.tensor(dct['data'])
      correct = dct['target']
      images.append(img)
      labels.append(correct)

    c = len(images)+1
    r = len(target_layers)+2
    fig = plt.figure(figsize=(30,14))
    fig.subplots_adjust(hspace=0.01, wspace=0.01)
    ax = plt.subplot(r, c, 1)
    ax.text(0.3,-0.5, "INPUT", fontsize=14)
    plt.axis('off')
    for i in range(len(target_layers)):
      target_layer = target_layers[i]
      ax = plt.subplot(r, c, c*(i+1)+1)
      ax.text(0.3,-0.5, target_layer, fontsize=14)
      plt.axis('off')

      for j in range(len(images)):
        image_cpu = images[j].cpu().numpy().astype(dtype=np.float32)
        for k in range(image_cpu.shape[0]):
          image_cpu[k] = image_cpu[k] * std[k] + mean[k]
        image_cpu = np.transpose(image_cpu, (1,2,0))
        img = np.uint8(255*image_cpu)
        if i==0:
          ax = plt.subplot(r, c, j+2)
          ax.text(0, 0.2, f"actual: {class_names[labels[j]]} \npredicted: {class_names[predicted[j][0]]}", fontsize=12)
          plt.axis('off')
          plt.subplot(r, c, c+j+2)
          plt.imshow(img)
          plt.axis('off')
          
        
        heatmap = 1-gcam_layers[i][j].cpu().numpy()[0] # reverse the color map
        heatmap = np.uint8(255 * heatmap)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        superimposed_img = cv2.resize(cv2.addWeighted(img, 0.5, heatmap, 0.5, 0), (128,128))
        plt.subplot(r, c, (i+2)*c+j+2)
        plt.imshow(superimposed_img, interpolation='bilinear')
        
        plt.axis('off')
    plt.show()
     



