import torch
from utils import sample_multivariate_distribution

def main():
    
    inputs = torch.ones((1,1,16,16))
    print(inputs.shape)
    conv1 = torch.nn.Conv2d(in_channels=1, out_channels=32, kernel_size=5)
    inputs = conv1(inputs)
    print(inputs.shape)
    conv2 = torch.nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=2)
    inputs = conv2(inputs)
    print(inputs.shape)
    conv3 = torch.nn.Conv2d(in_channels=64, out_channels=128, kernel_size=5)
    inputs = conv3(inputs)
    print(inputs.shape)

    deconv4 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=5)
    inputs = deconv4(inputs)
    print(inputs.shape)
    deconv5 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=4, stride=2)
    inputs = deconv5(inputs)
    print(inputs.shape)
    deconv6 = torch.nn.ConvTranspose2d(in_channels=32, out_channels=1, kernel_size=5)
    inputs = deconv6(inputs)
    print(inputs.shape)

    mean = torch.zeros(5, 128)
    var = torch.ones(5, 128)
    print(sample_multivariate_distribution(mean, var).size())





if __name__ == "__main__":
    main()