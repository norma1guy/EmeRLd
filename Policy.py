import torch
from Encoders import StateEncoder

class Actor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = StateEncoder().to(device=device)

        self.actor_net = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,7)
        )

    def forward(self, td):
        x = self.encoder(td)
        logits = self.actor_net(x)
        return logits

class Critic(torch.nn.Module):

    def __init__(self):
        super().__init__()
        
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.encoder = StateEncoder().to(device=device)

        self.critic_net = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,1)
        )

    def forward(self, td):
        x = self.encoder(td)
        state_value = self.critic_net(x)
        return state_value