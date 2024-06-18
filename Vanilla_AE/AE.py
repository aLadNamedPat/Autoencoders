import torch
import torch.nn.functional as F
import torch.nn as nn

#LOTS of help from tutorial here: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py

class AE(nn.Module):
    def __init__(
        self, 
        input_channels : int, 
        out_channels : int, 
        hidden_dims : list
        ) -> None:

        super(AE, self).__init__()

        encoder_store = []

        encoder_store.append(
            self.encoder_layer(input_channels, hidden_dims[0])
        )

        #Build a densely connected encoder with many skip connections
        for i in range(len(hidden_dims) - 1):
            encoder_store.append(
                self.encoder_layer(hidden_dims[i],
                                    hidden_dims[i + 1])
            )

        self.encoder = nn.Sequential(
            *encoder_store
        )

        decoder_store = []

        hidden_dims.reverse() #Reverse the hidden state list

        for i in range(len(hidden_dims) - 1):
            decoder_store.append(
                self.decoder_layer(
                    hidden_dims[i],
                    hidden_dims[i + 1]
                )
            )

        self.decoder = nn.Sequential(*decoder_store)

        self.fl = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1],
                hidden_dims[-1],
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding= 1
            ),
            nn.BatchNorm2d(
                hidden_dims[-1]
            ),
            nn.LeakyReLU(),
            nn.Conv2d(
                hidden_dims[-1],
                out_channels,
                kernel_size = 3,
                padding = 1
            ),
            nn.Tanh()
        )

    def _get_flattened_size(self, input_channels):
        with torch.no_grad():
            dummy_input = torch.zeros(1, input_channels, 64, 64)  # Example size, adjust if needed
            dummy_output = self.encoder(dummy_input)
            return torch.flatten(dummy_output, start_dim=1).size(1)
        
    def encoder_layer(
        self,
        input_channels : int,
        output_channels : int,
        ):
        return nn.Sequential(
                nn.Conv2d(
                    input_channels,
                    output_channels,
                    kernel_size= 3,
                    stride = 2,
                    padding = 1
                ), 
                nn.BatchNorm2d(output_channels),
                nn.LeakyReLU()
        )
    
    def decoder_layer(
        self,
        input_channels : int,
        output_channels : int,
    ):
        #Use ConvTranspose2d to upsample back to the original image size
        return nn.Sequential(
            nn.ConvTranspose2d(
                input_channels,
                output_channels,
                kernel_size = 3,
                stride = 2,
                padding = 1,
                output_padding= 1
            ),
            nn.BatchNorm2d(output_channels),
            nn.LeakyReLU()
        )

    def encode(
        self, 
        input : torch.Tensor
    ):
        r = self.encoder(input)
        # r = torch.flatten(r, start_dim = 1) #Flatten all dimensions of the encoded data besides the batch size
        # u = self.fc_mu(r)
        # var = self.fc_var(r)
        return r
        
    def decode(
        self,
        input : torch.Tensor,
    ):
        a = self.decoder(input)
        a =  self.fl(a)
        return a

    def forward(
        self,
        input : torch.Tensor,
    ):
        z = self.encode(input)
        a = self.decode(z)
        return a, input
    
    #Compute the loss of the encoder-decoder
    def find_loss(
        self,
        reconstructed,
        actual,
    ) -> int:
        recons_loss = F.mse_loss(reconstructed, actual)
        # recons_loss = nn.BCELoss(reconstructed, actual)
        # kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu  ** 2 - log_var.exp(), dim = 1), dim = 0)
        # loss = recons_loss + kld_loss * kld_weight
        return recons_loss
    