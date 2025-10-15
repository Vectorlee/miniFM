import os
import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from PIL import Image
from tqdm import tqdm

from fm_model import FlowMatching, FMConfig

model_dir = "model/"
mnist_dataset = datasets.MNIST(
    root = "./data",
    train = True,
    download = True, 
    transform = transforms.ToTensor()
)


epoch = 3
batch_size = 2048
device = "cuda" if torch.cuda.is_available() else 'cpu'

model = FlowMatching(FMConfig())
model = torch.compile(model)
model = model.to(device)
optim = torch.optim.AdamW(model.parameters(), lr=1e-4)

losses = []
for i in range(epoch):

    dataloader = DataLoader(mnist_dataset, batch_size=batch_size, shuffle=True)
    train_steps = len(dataloader) - 1
    data_iter = iter(dataloader)


    for step in range(train_steps):
        optim.zero_grad()

        images, labels = next(data_iter)
            
        B, C, H, W = images.shape      
        x1 = images
        x0 = torch.randn_like(images)
        target = x1 - x0

        t = torch.rand(batch_size, 1)
        t_c = t.unsqueeze(-1).unsqueeze(-1).expand(B, C, H, W)
        xt = (1 - t_c) * x0 + t_c * x1

        t = t.to(device)
        xt = xt.to(device)
        cls = labels.clone().view(batch_size, 1).type(torch.float32).to(device)
        logits = model(xt, t, cls)
            
        loss = ((target - logits)**2).mean()
        loss.backward()
        
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()

        if step % 20 == 0:
            print(f"step {step}, loss: {loss.item():.6f}, norm: {norm:.4f}")
        losses.append(loss.item())
    
    torch.save(model.state_dict(), os.path.join(model_dir, f"model_{i}.pth"))
