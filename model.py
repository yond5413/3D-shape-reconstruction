from encoder import Encoder
from decoder import Decoder
import torch
import torch.nn as nn
### for computational graph
from torchviz import make_dot
class Autoencoder(nn.Module):
    def __init__(self, latent_dim =100):
        super(Autoencoder, self).__init__()  
        self.latent_dim = latent_dim
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.decoder = Decoder(latent_dim=self.latent_dim)
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x 

if __name__ == "__main__":
    print("Just a test run to see if dimensions are correct")
    model = Autoencoder()
    input_shape = (3,256,256)
    tensor = torch.rand(1,*input_shape)
    batch_size = 4
    tensor_batched = tensor.repeat(batch_size, 1, 1, 1)

    with torch.no_grad():  # Disable gradient calculation during inference
        output = model(tensor_batched)#encoder(tensor)
        dot = make_dot(output, params=dict(model.named_parameters()))
        dot.render("model_graph")
    # Print the shape of the output vector
    print("Output shape:", output.shape)