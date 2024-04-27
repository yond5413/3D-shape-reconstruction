import torch
import torch.nn as nn
'''
TODO
encoder-> 2d input to get 3d info
decoder-> 3d info -> 3d reconstruction
encoder->> some cnn like resnet
decoder->> LTSM 
'''
## -> conv3dtranspose????

class Decoder(nn.Module):
     def __init__(self, latent_dim =100):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        ############################
        self.latent_dim = latent_dim
        # Transpose convolution layers
        self.conv1 = nn.ConvTranspose3d(latent_dim, 512, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv2 = nn.ConvTranspose3d(512, 256, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv3 = nn.ConvTranspose3d(256, 128, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv4 = nn.ConvTranspose3d(128, 64, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv5 = nn.ConvTranspose3d(64, 32, kernel_size=4, stride=2, padding=1, bias=False)
        #############################self.conv5 = nn.ConvTranspose3d(64, 1, kernel_size=4, stride=2, padding=1, bias=False)
        self.conv6 = nn.ConvTranspose3d(32, 16, kernel_size=4, stride=2, padding=1, bias=False)  # Upsample to 64x64x64
        self.conv7 = nn.ConvTranspose3d(16, 8, kernel_size=4, stride=2, padding=1, bias=False)   # Upsample to 128x128x128
        self.conv8 = nn.ConvTranspose3d(8, 4, kernel_size=4, stride=2, padding=1, bias=False)    # Upsample to 256x256x256
        self.conv9 = nn.ConvTranspose3d(4, 1, kernel_size=4, stride=1, padding=0, bias=False)     # No further upsampling, adjust kernel and stride
        # Activation function
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
     def forward(self,x):
        # Input shape: (batch_size, latent_dim)
        x = x.view(-1, self.latent_dim, 1, 1, 1)  # Reshape to (batch_size, latent_dim, 1, 1, 1)
        # Apply transpose convolutions
        x = self.relu(self.conv1(x))  # Output shape: (batch_size, 512, 2, 2, 2)
        x = self.relu(self.conv2(x))  # Output shape: (batch_size, 256, 4, 4, 4)
        x = self.relu(self.conv3(x))  # Output shape: (batch_size, 128, 8, 8, 8)
        x = self.relu(self.conv4(x))  # Output shape: (batch_size, 64, 16, 16, 16)
        x = self.relu(self.conv5(x))  # Output shape: (batch_size, 1, 32, 32, 32)
     #  x = self.conv5(x)
        x = self.relu(self.conv6(x))  # Output shape: (batch_size, 16, 64, 64, 64)
        x = self.relu(self.conv7(x))  # Output shape: (batch_size, 8, 128, 128, 128)
        x = self.relu(self.conv8(x))  # Output shape: (batch_size, 4, 256, 256, 256)
        x = self.conv9(x)
        x = self.sigmoid(x)
        ### cropping?
        x = x[:, :, :256, :256, :256]
        
        return x

if __name__ == "__main__":
    print("Just a test run to see if dimensions are correct")
    latent_dim = 100 
    decoder = Decoder(latent_dim)
    input_shape = (latent_dim)
    tensor = torch.rand(1,latent_dim)#*input_shape)
    print(tensor.size())
    with torch.no_grad():  # Disable gradient calculation during inference
        output = decoder(tensor)#tensor_batched)#encoder(tensor)

    # Print the shape of the output vector
    print("Output shape:", output.shape)