#had helped from https://pyimagesearch.com/2021/11/01/training-an-object-detector-from-scratch-in-pytorch/
import torch
import os
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor

#get device for training
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
print(f"using {device} device")

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        #nn.Sequential is an oredered container of modules, each executed in order explicitly specified below
        self.linear_reu_stack = nn.Sequential(
            nn.Linear(28*28, 512),  #28*28 is the size of the image, 
                                    #linear layer performs a linear transformation on the input data using stored weights and biases
            nn.ReLU(),  #ReLU introduces non-linear activations, which can lead to complex models
                        #Non-linear acitvations layers are usually added after linear layers
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )
        
    def forward(self, x):
        
        #flatten the image
        x = self.flatten(x)
        logits = self.linear_reu_stack(x)
        return logits


class BombDataset(torch.utils.data.Dataset):
    
    #for trasnform here we can utilise the CSGOImageProcessor class (as lambda) in addition to the ToTensor() function
    def __init__(self, root, transform=ToTensor()):
        self.root = root
        self.transform = transform
        self.images = list(sorted(os.listdir(os.path.join(root, "images"))))
        self.masks = list(sorted(os.listdir(os.path.join(root, "masks"))))
        
    def __getitem__(self, idx):
        img_path = os.path.join(self.root, "images", self.images[idx])
        mask_path = os.path.join(self.root, "masks", self.masks[idx])
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)
        mask = np.array(mask)
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[1:]
        masks = mask == obj_ids[:, None, None]
        num_objs = len(obj_ids)
        boxes = []
        for i in range(num_objs):
            pos = np.where(masks[i])
            xmin = np.min(pos[1])
            xmax = np.max(pos[1])
            ymin = np.min(pos[0])
            ymax = np.max(pos[0])
            boxes.append([xmin, ymin, xmax, ymax])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.ones((num_objs,), dtype=torch.int64)
        masks = torch.as_tensor(masks, dtype=torch.uint8)
        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        if self.transform is not None:
            img = self.transform(img)
        return img, target
    
    def __len__(self):
        return len(self.images)

model = NeuralNetwork().to(device)
print(model)
#calling model return 2d tensor with dim=0 corr to each output of 10 raw predicted values for each class (Defuse/NoDefuse)
#and dim=2 being the individual values of each output.
#can get prediction probability fby passing it through an instance of the nn.Softmax() class module
#Softmax layer is used to normalize the output of a network to a probability distribution over predicted output classes [0:1]
#for nn.Softmax(dim=1), dim indicates the dimension along which the values will sum to 1

#DEALING WITH MODEL PARAMETERS
#Many layers in a nn module are parameterized, meaning that they have associated weights and biases that are optimized during training
#nn module keeps track of these parameters in its parameters() method
#accessed as such 

print(f"Model structure: {model}\n\n")

for name, param in model.named_parameters():
    print(f"Layer: {name} | Size: {param.size()} | Values : {param[:2]} \n")



#Autograd Differentiation with Torch.AutoGrad
#When training nn, back propagation is used to update the weights of the model usually
#In backward propagation, the gradients of the loss function with respect to each parameter are used to adjust the parameters(model weights)
#To computer these gradients, we use autograd, which is a built-in differentiation engine that computes the gradient of a tensor
#autograd supports automatic differentiation for all operations on Tensors/ computation graphs

#EG
import torch

x = torch.ones(5)  # input tensor
y = torch.zeros(3)  # expected output
w = torch.randn(5, 3, requires_grad=True)
b = torch.randn(3, requires_grad=True)
z = torch.matmul(x, w)+b
loss = torch.nn.functional.binary_cross_entropy_with_logits(z, y)
#here w and b are parameters we want to optimise, so we set requires_grad=True bc we want to be able to compute gradients of loss func
#wrt to them -- imagine simple dl/dx = dl/dp * dp/dx, p being the parameter in qn
#can be done later using x.requires_grad_(True)

#a function applied to tensor to compute computation graph is in fact an object class Function
#this Function object knows hows to compute the function in the forward direction, and also 
#how to compute its derivative in the backward direction. Each tensor has a .grad_fn attribute that references the Function object

#COMPUTING GRADIENTS OF LOSS FUNCTION WRT TO PARAMETERS
#NAMELY: dloss/dw and dloss/db in a simple context
#computation done by calling loss.backward() and then accessing the gradient of w and b with respect to the loss
loss.backward()
print(w.grad)
print(b.grad)

#DISABLING GRADIENT TRACKING WHEN THERES NO NEED
#All tensor by default have requires_grad=True, which means that all operations on them are tracked by autograd
#but in cases where we dont need to track the gradients, we can wrap the code block in with torch.no_grad()
#EG of cases, applying model to some input data (only want forward pass), not training model
with torch.no_grad():
    z = torch.matmul(x, w)+b
    print(z.requires_grad)

#or 
z = torch.matmul(x, w)+b
z_Det = z.detach()
print(z_Det.requires_grad)

#REASON TO DISABLE GRADIENT TRACKING
# 1) to reduce memory usage and speed up computations
# 2) to freeze some parameters of the model during finetuning a pretrained model as we dont want to backprop through them


#
loss_fn = nn.CrossEntropyLoss()
optimizers = torch.optim.SGD(model.parameters(), lr=1e-3)

def train(dataloader, model, loss_fn, optimizers):
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        pred = model(X)
        test_loss += loss_fn(pred, y)
        correct += (pred.argmax(1) == y).type(torch.float).sum().item()
    test_loss /= num_batches
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.1f}%  , Avg loss: {test_loss:>8f} \n")