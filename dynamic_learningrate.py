import torch

def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))#每30个epoch，学习率变为原来的1/10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
        
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9)
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)