import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim, num_layers, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.dropout = nn.Dropout(0.2)
        self.net = nn.Sequential()
        init_fxn = nn.init.xavier_uniform_
        for i in range(num_layers):
            act = nn.ReLU()
            if num_layers == 1:
                layer = nn.Linear(in_dim, out_dim)
                init_fxn(layer.weight)
                self.net.append(layer)
            elif i == 0:
                layer = nn.Linear(in_dim, hidden_dim)
                init_fxn(layer.weight)
                self.net.append(layer)
                self.net.append(act)
                self.net.append(nn.Dropout(0.2))
            elif i < num_layers - 1:
                layer = nn.Linear(hidden_dim, hidden_dim)
                init_fxn(layer.weight)
                self.net.append(layer)
                self.net.append(act)
                self.net.append(nn.Dropout(0.2))
            else:
                layer = nn.Linear(hidden_dim, out_dim)
                init_fxn(layer.weight)
                self.net.append(layer)


    def forward(self, x):
        return self.net(self.dropout(x))