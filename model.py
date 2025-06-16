import torch
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F

class MalwareNetSmall(nn.Module):

    def __init__(self, input_size=115, hidden_dims=[256, 128, 64], num_classes=10, dropout_rate=0.5):
        super(MalwareNetSmall, self).__init__()
        
        # Wider network with residual connections for better feature extraction
        self.layer1 = nn.Sequential(
            nn.Linear(input_size, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.layer2 = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[1]),
            nn.BatchNorm1d(hidden_dims[1]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )

        self.layer3 = nn.Sequential(
            nn.Linear(hidden_dims[1], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate)
        )
        
        # Additional layer for better feature extraction
        self.layer4 = nn.Sequential(
            nn.Linear(hidden_dims[2], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2]),
            nn.LeakyReLU(0.1),
            nn.Dropout(dropout_rate * 0.5)  # Lower dropout before final layer
        )
        
        # Shortcut connection from layer 1 to final layer
        self.shortcut = nn.Sequential(
            nn.Linear(hidden_dims[0], hidden_dims[2]),
            nn.BatchNorm1d(hidden_dims[2])
        )

        self.output = nn.Linear(hidden_dims[2], num_classes)
        
        # Initialize weights for better convergence
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm1d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)    def forward(self, x):
        # Main path
        x1 = self.layer1(x)
        x = self.layer2(x1)
        x = self.layer3(x)
        
        # Residual connection
        shortcut = self.shortcut(x1)
        
        # Add residual connection
        x = self.layer4(x) + shortcut
        
        # Final classification layer
        x = self.output(x)
        return x

