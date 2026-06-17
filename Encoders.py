import torch.nn as nn
import torch

class PlayerPokemonEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.MAX_SPECIES = 412
        self.MAX_MOVES = 355
        self.MAX_ABILITIES = 78
        self.MAX_ITEMS = 377
        self.MAX_STAT1 = 6

        # Embeddings 
        self.species_emb = nn.Embedding(self.MAX_SPECIES,16)
        self.move_emb = nn.Embedding(self.MAX_MOVES,16)
        self.ability_emb = nn.Embedding(self.MAX_ABILITIES,16)
        self.status_emb = nn.Embedding(self.MAX_STAT1,16)
        self.hold_item_emb = nn.Embedding(self.MAX_ITEMS,16)

        self.player_mlp = nn.Sequential(
            nn.Linear( 137,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

    def forward(self,playerpokemon):
        
        species = self.species_emb(playerpokemon['species'])
        moves = self.move_emb(playerpokemon['moves'])
        ability = self.ability_emb(playerpokemon['ability'])
        status = self.status_emb(playerpokemon['status1'])
        item = self.hold_item_emb(playerpokemon['holdItem'])
        moves = moves.flatten(start_dim=-2)
        
        num_stats = torch.cat([
            playerpokemon['hp'].unsqueeze(-1),
            playerpokemon['lvl'].unsqueeze(-1)/100.0,
            playerpokemon['att'].unsqueeze(-1).float(),
            playerpokemon['defn'].unsqueeze(-1).float(),
            playerpokemon['spe'].unsqueeze(-1).float(),
            playerpokemon['spA'].unsqueeze(-1).float(),
            playerpokemon['spD'].unsqueeze(-1).float(),
            playerpokemon['pp'].float().mean(dim=-1, keepdim=True),
            playerpokemon['exp'].unsqueeze(-1).float(),

        ], dim=-1)
        
        x = torch.cat([species,moves,ability,status,item,num_stats], dim=-1)

        return self.player_mlp(x)
    

class EnemyPokemonEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        self.MAX_STAT1 = 6
        self.stat_emb = nn.Embedding(self.MAX_STAT1,16)

        self.enemy_mlp = nn.Sequential(
            nn.Linear(16 + 2,128),
            nn.ReLU(),
            nn.Linear(128,128)
        )

    def forward(self,enemypokemon) :

        status = self.stat_emb(enemypokemon['status1'])
        rest = torch.cat([
            enemypokemon['hp'].unsqueeze(-1),
            enemypokemon['lvl'].unsqueeze(-1)/100.0,

        ],dim=-1)

        x = torch.cat([status,rest],dim=-1)

        return self.enemy_mlp(x)
    
class PixelEncoder(nn.Module) :

    def __init__(self,output_dim=512) :
        super().__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels=3,
                out_channels=32,
                kernel_size=8,
                stride=4
            ),
            nn.ReLU(),
            nn.Conv2d(
                in_channels=32,
                out_channels=64,
                kernel_size=4,
                stride=2
            )
            ,
            nn.ReLU(),
            nn.Conv2d(
                in_channels=64,
                out_channels=64,
                kernel_size=3,
                stride=1
            )
        )
        
        with torch.no_grad() :
            dummy = torch.zeros(1,3,160,240)
            conv_out = self.conv(dummy)
            conv_size = conv_out.flatten(1).shape[1]

        self.fc = nn.Linear(conv_size,output_dim)

    def forward(self, x):
        if x.shape[-4] == 1:
            x = x.squeeze(-4)

        batch_shape = x.shape[:-3]

        x = x.flatten(0, -4)
        x = self.conv(x)
        x = x.flatten(1)
        x = torch.relu(self.fc(x))
        x = x.unflatten(0, batch_shape)

        return x


class StateEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.token_dim = 256

        #Battle Transformer
        self.player_proj = nn.Linear(128, self.token_dim)
        self.enemy_proj = nn.Linear(128, self.token_dim)
        self.side_embedding = nn.Embedding(2, self.token_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=256,
            nhead=8,
            batch_first=True
        )

        self.pokemon_transformer = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2
        )
        self.time_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=256,
                nhead=8,
                batch_first=True
            ),
            num_layers=2
        )

        self.player_encoder = PlayerPokemonEncoder()
        self.enemy_encoder = EnemyPokemonEncoder()
        self.pixel_encoder = PixelEncoder()
        self.party_encoder = nn.Linear(2, 128)
        self.global_mlp = nn.Sequential(
            nn.Linear(12 + 8 * 2,32),
            nn.ReLU(),
        )
        self.final = nn.Sequential(
            nn.Linear(128 * 3 + 8 + 32 + 512,256),
            nn.ReLU(),
        )

    def forward(self,state) :

        player = self.player_proj(self.player_encoder(state['playerpokemon']))
        enemy = self.enemy_proj(self.enemy_encoder(state['enemypokemon']))
        player_tokens = (player + self.side_embedding.weight[0])
        enemy_tokens = (enemy + self.side_embedding.weight[1])
        pokemon_tokens = torch.cat([player_tokens, enemy_tokens],dim=-2)
        
        
        orig_shape = pokemon_tokens.shape[:-2]

        N = pokemon_tokens.shape[-2]
        D = pokemon_tokens.shape[-1]

        pokemon_tokens = pokemon_tokens.reshape(-1, N, D)

        pokemon_tokens = self.pokemon_transformer(
            pokemon_tokens
        )

        battle_repr = pokemon_tokens.mean(dim=1)

        battle_repr = battle_repr.reshape(
            *orig_shape,
            D
        )
        battle_repr = self.time_transformer(battle_repr)

        pixels = self.pixel_encoder(state['pixelbuffer'])
        pixels = pixels.squeeze(1)

        partyhp = state['party']['hp'].mean(dim=-1, keepdim=True)
        partylvl = (state['party']['level'].float()/100.0).mean(dim=-1, keepdim=True)
        party = torch.cat([partyhp,partylvl], dim=-1)
        party = self.party_encoder(party)
        storyflags = torch.cat([v.float() for _, v in state["storyflags"].items()],dim=-1)

        map = state['map']
        global_vecs = torch.cat([
            state['battleactioncursor'].float(),
            state['battlemovecursor'].float(),
            state['battleoutcome'].float(),
            state['battletype'].float(),
            state['textactive'].float(),
            state['textstate'].float(),
            state['badge'].float(),
            state['hms'].float(),
            state['startmenu'].float(),
            storyflags,
        ], dim=-1)
        #print(global_vecs)
        global_vecs = self.global_mlp(global_vecs)
    
        #print("battle_repr", battle_repr.shape)
        #print("party", party.shape)
        #print("map", map.shape)
        #print("pixels", pixels.shape)
        #print("global_vecs", global_vecs.shape)
        x = torch.cat([battle_repr,party,map,pixels,global_vecs], dim=-1)
        #print(state['inbattle'])

        return [self.final(x),state['inbattle']]





