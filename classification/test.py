import torchxrayvision as xrv
import torch

# model = xrv.models.DenseNet(num_classes=3)
#
# x = torch.randn(1, 1, 512, 512)
#
# out = model(x)
#
# print(out, out.shape)


x = torch.tensor([[1., 0.6, 0.]])

y = torch.tensor([[1., 0., 0.]])

criteon = torch.nn.BCELoss()


loss = criteon(x, y)

print(loss)