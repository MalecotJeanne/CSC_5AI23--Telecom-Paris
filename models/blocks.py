import torch.nn as nn

##### ENCODER BLOCK #####

class EncoderBlock(nn.Module):
    def __init__(self, n_channels, channels, act = nn.LeakyReLU):
        super(EncoderBlock, self).__init__()

        self.in_channels = [n_channels] + list(channels[:-1])
        self.out_channels = list(channels)
        
        self.act = act
        
        self.blocks = nn.ModuleList()
        for i in range(len(self.in_channels) - 1):
            self.blocks.append(nn.Sequential(
                            nn.Conv2d(self.in_channels[i], self.out_channels[i], kernel_size = 3, stride=2, padding=1),
                            self.act(),
                            ))
        self.blocks.append(nn.Sequential(
                        nn.Conv2d(self.in_channels[-1], self.out_channels[-1], kernel_size = 1, stride=1, padding = "same"),
                         ))  
        
    def forward(self, x):

        for block in self.blocks:
            x = block(x)

        return x

##### DECODER BLOCK #####

class DecoderBlock(nn.Module):
    def __init__(self, n_channels, channels, act = nn.LeakyReLU):
        super(DecoderBlock, self).__init__()

        reverse_channels = list(channels)
        reverse_channels.reverse()
        
        self.in_channels = reverse_channels
        self.out_channels = reverse_channels[1:] + [n_channels]
        
        self.act = act

        self.blocks = nn.ModuleList()
        self.blocks.append(nn.Sequential(   
                        nn.Conv2d(self.in_channels[0], self.out_channels[0], kernel_size = 1, stride=1, padding = 0),
                        self.act(),
                        ))
        for i in range(1,len(self.in_channels)):
            self.blocks.append(nn.Sequential(
                            nn.ConvTranspose2d(self.in_channels[i], self.out_channels[i], kernel_size = 2, stride=2, padding = 0),
                            self.act(),
                            ))
    def forward(self, x):
        
        for block in self.blocks:
            x = block(x)
            
        return x
        
        