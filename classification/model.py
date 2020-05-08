from torch import nn
from torchvision import models
import torch



class covid_net(nn.Module):
    def __init__(self, name='resnet18'):
        super(covid_net, self).__init__()
        model = self.get_base_model(name = name)


        self.base_model = nn.Sequential(*list(model.children())[:-1])
        #
        in_size = 0
        if 'densenet' in name:
            in_size = 1024 * 16 * 16
        else: in_size = model.fc.in_features
        self.cf1 = nn.Linear(in_features=in_size, out_features=3)
    def forward(self, x):
        x = self.base_model(x)
        # print(x.shape)
        x = x.view(-1, torch.prod(torch.tensor(x.shape[1:])).item())
        x = self.cf1(x)
        x = torch.sigmoid(x)
        return x

    def get_base_model(self, name='resnet18'):
        model = models.resnet18(pretrained=True)
        if name == 'resnet18':
            model = models.resnet18(pretrained=True)
        if name == 'resnet50':
            model = models.resnet50(pretrained=True)
        if name == 'resnet101':
            model = models.resnet101(pretrained=True)

        if name == 'densenet121':
            model = models.densenet121(pretrained=True)

        if name == 'googlenet':
            model = models.googlenet(pretrained=True)

        return model

def main():
    # tset

    x = torch.randn(4, 3, 512, 512)
    model = covid_net(name='densenet121')
    out = model(x)

    print(out, out.shape)
    pass
if __name__ == '__main__':
    main()