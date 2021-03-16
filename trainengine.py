import torch 
from tqdm import tqdm




def train(loader, model, optimization, device, scheduler):
    model.train()
    loss_final = 0
    for d in tqdm(loader, total=len(loader)):
        for x,y in d.items():
            d[x] = y.to(device)
        optimization.zero_grad()
        _, loss = model(**d)
        loss.backward()
        optimization.step()
        scheduler.step()
        loss_final += loss.item()
    return loss_final / len(loader)


def eval(loader, model, device):
    model.eval()
    loss_final = 0
    for d in tqdm(loader, total=len(loader)):
        for x,y in d.items():
            d[x] = y.to(device)
        _, loss = model(**d)
        loss_final += loss.item()
    return loss_final / len(loader)