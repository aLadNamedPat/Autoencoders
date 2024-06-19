import torch
import torch.nn.functional as F
import torch.nn as nn

#LOTS of help from tutorial here: https://github.com/AntixK/PyTorch-VAE/blob/master/models/vanilla_vae.py


class VAE(nn.Module):
    def __init__(
        self, 
        input_channels : int, 
        out_channels : int, 
        latent_dim : int, 
        hidden_dims : list
        ) -> None:

        super(VAE, self).__init__()

        self.encoder_store = []

        self.encoder_store.append(
            self.encoder_layer(input_channels, hidden_dims[0])
        )

        #Build a densely connected encoder with many skip connections
        for i in range(len(hidden_dims) - 1):
           self.encoder_store.append(
                self.encoder_layer(hidden_dims[i],
                                    hidden_dims[i + 1])
            )

        self.encoder = nn.Sequential(
            *self.encoder_store
        )

        self.fc_mu = nn.Linear(hidden_dims[-1]*16, latent_dim) #VAE mean calculated
        self.fc_var = nn.Linear(hidden_dims[-1]*16, latent_dim) #VAE variance calculated

        self.decoder_store = []

        #Need to reverse the encoder process to build the decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1]*16)

        hidden_dims.reverse() #Reverse the hidden state list

        for i in range(len(hidden_dims) - 1):
            self.decoder_store.append(
                self.decoder_layer(
                    hidden_dims[i] * 2,
                    hidden_dims[i + 1]
                )
            )

        self.decoder = nn.Sequential(*self.decoder_store)

        self.fl = nn.Sequential(
            nn.ConvTranspose2d(
                hidden_dims[-1] * 2,
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
            dummy_input = torch.zeros(1, input_channels, 4, 4)  # Example size, adjust if needed
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
        skip_connections = []
        for layer in self.encoder_store:
            input = layer(input)
            skip_connections.append(input)  #Append all the inputs from the previous layers [128, 128, 256, 512, 512]

        r = torch.flatten(input, start_dim = 1) #Flatten all dimensions of the encoded data besides the batch size
        u = self.fc_mu(r)
        var = self.fc_var(r)
        return u, var, skip_connections
    
    def reparamterize(
        self,
        u,
        var
    ): #Leveraging the reparametrization trick from the original VAE paper
        std = torch.exp(0.5 * var)
        eps = torch.randn_like(std)  #reparametrization trick implemented here!
        return eps * std + u #we separate the randomization of z from mu and std which makes it differentiable
    
    def decode(
        self,
        input : torch.Tensor,
        l_hidden_dim : int,
        skip_connections : list
    ):
        a = self.decoder_input(input)
        a = a.view(-1, l_hidden_dim, 4, 4)
        skip_connections = skip_connections[::-1]
        for i, layer in enumerate(self.decoder_store):
            a = torch.cat((a, skip_connections[i]), dim = 1)
            a = layer(a)
        a = torch.cat((a, skip_connections[-1]), dim = 1)
        a =  self.fl(a)
        return a

    def forward(
        self,
        input : torch.Tensor,
    ):
        u, var, skip_connections = self.encode(input)
        z = self.reparamterize(u, var)
        a = self.decode(z, 512, skip_connections)
        return a, input, u, var
    
    #Compute the loss of the encoder-decoder
    def find_loss(
        self,
        reconstructed,
        actual,
        mu,
        log_var,
        kld_weight
    ) -> int:
        recons_loss = F.mse_loss(reconstructed, actual)
        # recons_loss = nn.BCELoss(reconstructed, actual)
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu  ** 2 - log_var.exp(), dim = 1), dim = 0)
        loss = recons_loss + kld_loss * kld_weight
        return loss