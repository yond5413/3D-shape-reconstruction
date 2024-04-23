import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels,kernel_size, stride, padding):
        super(ResidualBlock,self).__init__()
        self.conv1 = nn.Conv2d(in_channels,out_channels,kernel_size, stride,padding,bias = False)
        self.conv2 = nn.Conv2d(out_channels,out_channels,kernel_size, stride=1 ,padding=padding, bias = False)
        self.relu  = nn.ReLU(out_channels)
        self.batchNorm  = nn.BatchNorm2d(out_channels)
        if stride != 1:
            self.down_sample = nn.Conv2d(in_channels,out_channels,kernel_size=(1,1), stride=stride, padding=0, bias = False)
        else:
            self.down_sample =None       
    def forward(self,x):
        identity = x
        out1 = self.conv1(x)
        f = self.relu(self.batchNorm(out1))
        #################
        f = self.conv2(f)
        #################
        if self.down_sample:    
            identity = self.down_sample(identity)
        h = f+identity
        h  = self.batchNorm(h)
        ret = self.relu(h) 
        return ret
##############################
#### can potentially use 3d version of resnet-18 for encoder
### -> essentially same as what was used in hpml hw2/cv2 hw3
### but removed fc layer for 
#### just have to remove fc/out/linear layers in encoder
class Encoder(nn.Module):
    def __init__(self, in_channels=3, latent_dim=100):
        super(Encoder, self).__init__()
        self.input_layer = nn.Conv2d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block1_b = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1)
        self.block2 = ResidualBlock(in_channels=64, out_channels=64, kernel_size=3, stride=2, padding=1)
        self.block2_b = ResidualBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2, padding=1)
        self.block3 = ResidualBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.block3_b = ResidualBlock(in_channels=256, out_channels=256, kernel_size=3, stride=2, padding=1)
        self.block4 = ResidualBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.block4_b = ResidualBlock(in_channels=512, out_channels=512, kernel_size=3, stride=2, padding=1)
        self.latent_dim = latent_dim
        self.output_layer = nn.Linear(in_features=8192, out_features=latent_dim)#nn.Linear(in_features=512, out_features=latent_dim)
    def forward(self, x):
        out1 = self.block1(self.input_layer(x))
        out1_b = self.block1_b(out1)
        out2 = self.block2(out1_b)
        out2_b = self.block2_b(out2)
        out3 = self.block3(out2_b)
        out3_b = self.block3_b(out3)
        out4 = self.block4(out3_b)
        out4_b = self.block4_b(out4)
        y = out4_b.view(out4_b.size(0), -1)  # Flatten
        representation = self.output_layer(y)
        return representation
    
if __name__ == "__main__":
    print("Just a test run to see if dimensions are correct")
    encoder = Encoder()
    input_shape = (3,256,256)
    tensor = torch.rand(1,*input_shape)
    batch_size = 4
    tensor_batched = tensor.repeat(batch_size, 1, 1, 1)

    with torch.no_grad():  # Disable gradient calculation during inference
        output = encoder(tensor_batched)#encoder(tensor)

    # Print the shape of the output vector
    print("Output shape:", output.shape)