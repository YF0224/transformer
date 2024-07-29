import torch
aa = torch.tensor([1, 2, 3]).cuda()
if torch.cuda.is_available():
    device = torch.device("cuda")
    x = torch.randn(10, 10).to(device)  # 创建一个随机张量并移动到GPU
    y = torch.randn(10, 10).to(device)
    z = x + y  # 在GPU上进行计算
    print(z)