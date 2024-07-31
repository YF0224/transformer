from model import Transformer
from torch.utils.data import DataLoader
from data import MyDataset

my_model = Transformer()
dataset = MyDataset("source.txt", "target.txt")
dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

for input_id, input_m, output_id, output_m in dataloader:
    my_model(input_id, input_m, output_id[:, :-1], output_m[:, :-1])
