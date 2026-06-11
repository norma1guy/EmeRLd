import torch
from Encoders import StateEncoder

class Actor(torch.nn.Module):

    def __init__(self):
        super().__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.encoder = StateEncoder().to(device=device)

        self.overworld_net = torch.nn.Sequential(
            torch.nn.Linear(256, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128, 128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,7)
        )
        self.battle_net = torch.nn.Sequential(
            torch.nn.Linear(256,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,128),
            torch.nn.ReLU(),
            torch.nn.Linear(128,7)
        )

    def forward(self, td):
        x, inbattle = self.encoder(td)

        overworld_logits = self.overworld_net(x)
        battle_logits = self.battle_net(x)

        logits = torch.where(
            inbattle.bool(),
            battle_logits,
            overworld_logits
        )

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
        x,inbattle = self.encoder(td)
        state_value = self.critic_net(x)
        return state_value