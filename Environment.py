import posix_ipc
import mmap
import struct
from Memory import Memory
from Map import Map,MapNode
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec,UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from typing import Optional
import torch

class Environment(EnvBase) :

    def __init__(self,td_params=None,seed=None,device='cuda'):

        super().__init__(device=device,batch_size=[])
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        self.shm = posix_ipc.SharedMemory("/RAM_MAP")
        self.pyFlag = posix_ipc.Semaphore("/py_to_lua")
        self.luaFlag  = posix_ipc.Semaphore("/lua_to_py")
        self.ewramSize = 256 * 1024
        self.iwramSize = 32 * 1024
        self.mm = mmap.mmap(self.shm.fd,8 + self.ewramSize + self.iwramSize)
        self.shm.close_fd()
        self.input = struct.unpack("I",self.mm[:4])
        self.ewram = Memory(self.mm,8)
        self.iwram = Memory(self.mm,8 + self.ewramSize)
        self.is_battle = False
        self.max_battlers = 4
        self.double_battle = False
        self.map = Map().build_graph()
        self.move_decay = 0.99
        self.visited = {}
        self.ru8 = self.ewram.read_u8
        self.ru16le = self.ewram.read_u16_le
        self.ru16be = self.ewram.read_u16_be
        self.ru32le = self.ewram.read_u32_le
        self.ru32be = self.ewram.read_u32_be
        self.rs8 = self.ewram.read_s8
        self.rs16le = self.ewram.read_s16_le
        self.rs16be = self.ewram.read_s16_be
        self.rs32le = self.ewram.read_s32_le
        self.rs32be = self.ewram.read_s32_be
        self.iwru32le = self.iwram.read_u32_le
        self.party_hp = torch.zeros(6, device=self.device,dtype=torch.float32)
        self.party_level = torch.zeros(6, device=self.device, dtype=torch.int64)
        self.party_status = torch.zeros(6, device=self.device, dtype=torch.int64)

        self.p_att = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_defn = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_spe = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_spA = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_spD = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_exp = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_stat_changes = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_status = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_moves = torch.zeros(
                    (2, 4),
                    dtype=torch.int64,
                    device=self.device
                )
        self.p_pp = torch.zeros(
                    (2, 4),
                    dtype=torch.int64,
                    device=self.device
                )
        self.p_types = torch.zeros(
                    (2, 2),
                    dtype=torch.int64,
                    device=self.device
                )
        self.p_item = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_species = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.p_ability = torch.zeros(2, device=self.device, dtype=torch.int64)

        self.p_hp = torch.zeros(2, device=self.device)
        self.p_lvl = torch.zeros(2, device=self.device, dtype=torch.int64)

        self.e_hp = torch.zeros(2, device=self.device)
        self.e_lvl = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.e_status = torch.zeros(2, device=self.device, dtype=torch.int64)
        self.hms = torch.zeros(8, device=self.device, dtype=torch.int64)
        self.coords = torch.zeros(6, device=self.device, dtype=torch.int64)

        self.badges = torch.zeros(8, device=self.device, dtype=torch.int64)

        self.out_td = TensorDict({}, batch_size=[])

    def _get_state(self):
        self.is_battle = 1  if self.ru32le(0x22fec) != 0 else 0

        mapState = self.get_coords()
        #print(mapState)
        badges = self.get_badges()
        party = self.get_party()
        hms = self.get_hms()
        player_battle_state,enemy_battle_state = self.get_battle_info()

        out = TensorDict(
            {
                'inbattle' : torch.tensor([self.is_battle],dtype=torch.int64,device=self.device),
                'playerpokemon' : player_battle_state,
                'enemypokemon' : enemy_battle_state,
                'map' : mapState,
                'badge' : badges,
                'party' : party,
                'hms' : hms,
            },
            batch_size=[]
        )
        
        return out
    
    def _make_spec(self):

        self.observation_spec = CompositeSpec(

            observation = CompositeSpec(
                inbattle = DiscreteTensorSpec(2, shape=(1,), dtype=torch.int64),
                playerpokemon = CompositeSpec(
                    att = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    defn = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    spe = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    spA = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    spD = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    types = UnboundedDiscreteTensorSpec(shape=(2,2),dtype=torch.int64),
                    pp = UnboundedDiscreteTensorSpec(shape=(2,4),dtype=torch.int64),
                    statChanges = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    status1 = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    lvl = DiscreteTensorSpec(
                        n=101,
                        shape=(2,),
                        dtype=torch.int64,
                    ),
                    exp = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    hp = UnboundedContinuousTensorSpec(shape=(2,),dtype=torch.float32),
                    ability = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    moves = UnboundedDiscreteTensorSpec(shape=(2,4),dtype=torch.int64),
                    holdItem = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                    species = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),

                ),
                enemypokemon = CompositeSpec(
                    hp = UnboundedContinuousTensorSpec(shape=(2,),dtype=torch.float32),
                    lvl = DiscreteTensorSpec(
                        n=101,
                        shape=(2,),
                        dtype=torch.int64,
                    ),
                    status1 = UnboundedDiscreteTensorSpec(shape=(2,),dtype=torch.int64),
                ),
                map = UnboundedDiscreteTensorSpec(shape=(6,),dtype=torch.int64),
                badge = DiscreteTensorSpec(
                    n=2,
                    shape=(8,),
                    dtype=torch.int64,
                ),
                party = CompositeSpec(
                    status = UnboundedDiscreteTensorSpec(shape=(6,),dtype=torch.int64),
                    level = DiscreteTensorSpec(
                        n=101,
                        shape=(6,),
                        dtype=torch.int64,
                    ),
                    hp = UnboundedContinuousTensorSpec(shape=(6,),dtype=torch.float32),
                ),
                hms = DiscreteTensorSpec(
                    n=2,
                    shape=(8,),
                    dtype=torch.int64,
                ),
            )
        )

        self.action_spec = DiscreteTensorSpec(
            n=8,
            shape=(1,),
            dtype=torch.int64,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
        )

        self.done_spec = DiscreteTensorSpec(
            n=2,
            shape=(1,),
            dtype=torch.bool,
        )


    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def calc_reward(self,prevState,nextState) :
        reward = 0
        # Badge reward
        if prevState is None :
            pass
        elif nextState['badge'].sum() > prevState['badge'].sum() :
            reward += 10 
        
        # Movement reward
        if not torch.equal(nextState['map'],prevState['map']) :
            visited_flag = nextState['map'][3].item()
            new_place = nextState['map'][4].item()
            if visited_flag :
                reward += self.move_decay * 0.05 if new_place else self.move_decay * 0.01
                self.move_decay = 0.99
            else :
                self.move_decay = max(0.2, self.move_decay * 0.99)
                reward += self.move_decay * 0.01


        # HMs reward
        if prevState is None :
            pass
        elif nextState['hms'].sum() > prevState['hms'].sum() :
            reward += 10
        

        #Battle reward
        prev_hp = prevState['party']['hp'].sum()
        next_hp = nextState['party']['hp'].sum()

        prev_lvl = prevState['party']['level'].sum()
        next_lvl = nextState['party']['level'].sum()

        prev_status = prevState['party']['status']
        next_status = prevState['party']['status']

        # HP loss penalty
        if next_hp < prev_hp:
            reward -= 0.001 * (prev_hp - next_hp).item()

        # Status penalty
        if (next_status != 0).any():
            reward -= 0.01

        # Level reward
        if next_lvl > prev_lvl:
            reward += 0.001 * (next_lvl - prev_lvl).item()

        # Whiteout penalty
        if next_hp <= 0:
            reward -= 1.0

        # Healing reward
        if prev_hp <= 0 and next_hp > 0:
            reward += 0.5
                
        return torch.tensor(reward, dtype=torch.float32,device=self.device)
    
    def _step(self, tensordict):

        active, state = self._get_textbox_flags()

        action = tensordict.get('action')
        act_val = int(action)  

        prev_obs = tensordict['observation']  
        self.luaFlag.acquire()
        self.mm[:4] = struct.pack("I", act_val)
        next_obs = self._get_state()
        reward = self.calc_reward(prev_obs, next_obs)

        self.pyFlag.release()

        done = False
        reward = reward.unsqueeze(0)

        done = torch.tensor(
            [done],
            dtype=torch.bool,
            device=self.device
        )
        
        return TensorDict(
            {
                "observation": next_obs,
                "reward": reward,
                "done": done
            },
            batch_size=[]
        )
    
    def _reset(self, tensordict):

        self.luaFlag.acquire()

        self.mm[:4] = struct.pack('I', 8)

        self.pyFlag.release()

        if tensordict is None:
            tensordict = TensorDict({}, batch_size=[])

        tensordict['observation'] = self._get_state()

        return tensordict
            
    def _get_textbox_flags(self) :

        active = self.ru8(0x0201cb)
        state = self.ru8(0x0201cc)

        return (active,state)


    
    def get_hms(self):
        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000
        base = saveBlockAddr + 0x690 + 339

        read16 = self.ru16le  # cache function

        for i in range(8):
            self.hms[i] = 1 if read16(base + 4 * i) else 0

        return self.hms.clone()

   
    def get_coords(self):

        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000

        x = self.rs16le(saveBlockAddr)
        y = self.rs16le(saveBlockAddr + 0x2)
        direction = self.rs16le(0x037368) // 17
        mapId = self.ru16le(saveBlockAddr + 0x32)
        new_place = None
        if mapId not in self.visited:
            node = MapNode(mapId)
            self.visited[mapId] = node
            new_place = 1
        else:
            node = self.visited[mapId]
            new_place = 0

        visited = node.update_visit(x, y)

        self.coords[0] = x
        self.coords[1] = y
        self.coords[2] = direction
        self.coords[3] = mapId
        self.coords[4] = visited
        self.coords[5] = new_place

        return self.coords.clone()
    
    def get_party(self):

        partyCount = self.ru8(0x244e9)
        partyAddr = 0x244ec
        structSize = 104 # size of each Pokemon struct
        for i in range(6):
            if i < partyCount :
                pokemon = partyAddr + i * structSize
                status = self.ru32le(pokemon + 0x50)
                level = self.ru8(pokemon + 0x54)
                hp = self.ru16le(pokemon + 0x56)
                maxhp = self.ru16le(pokemon + 0x58)
                self.party_status[i] = status
                self.party_level[i] = level
                self.party_hp[i] = hp / maxhp
            else:
                self.party_status[i] = 0
                self.party_level[i] = 0
                self.party_hp[i] = 0

        out = TensorDict({
            'status': self.party_status,
            'hp': self.party_hp,
            'level': self.party_level,
            },
            batch_size=[]
        )
        return out.to(device=self.device)
    
    def get_badges(self):

        gym_flag = 0x867

        read_flag = self.ewram.read_flag 

        for i in range(8):
            self.badges[i] = 1 if read_flag(gym_flag) else 0
            gym_flag += 1

        return self.badges.clone()
    
    def get_battle_info(self):
        gBattleMons = 0x24084
        size = 88

        battlers = self.ru8(0x2406c)
        is_double = battlers > 2

        # reset tensors
        self.p_att.zero_()
        self.p_defn.zero_()
        self.p_spe.zero_()
        self.p_spA.zero_()
        self.p_spD.zero_()
        self.p_hp.zero_()
        self.p_lvl.zero_()

        self.e_hp.zero_()
        self.e_lvl.zero_()
        self.e_status.zero_()

        p_count = 0
        e_count = 0

        for i in range(battlers):
            mon = gBattleMons + i * size

            attack = self.ru16le(mon + 0x2)
            defense = self.ru16le(mon + 0x4)
            speed = self.ru16le(mon + 0x6)
            spA = self.ru16le(mon + 0x8)
            spD = self.ru16le(mon + 0xa)
            types = [self.ru8(mon + 0x21),self.ru8(mon + 0x22)]
            pp = [self.ru8(mon + 0x24 + i) for i in range(4)]
            statChanges = self.rs8(mon + 0x18)
            status1 = self.ru32le(mon + 0x4C)
            exp = self.ru32le(mon + 0x44)
            ability = self.ru8(mon + 0x20)
            moves = [self.ru16le(mon + 0x0C + 2 * i) for i in range(4)]
            holdItem = self.ru16le(mon + 0x2E)
            species = self.ru16le(mon)

            lvl = self.ru8(mon + 0x2A)
            hp = self.ru16le(mon + 0x28)
            maxhp = self.ru16le(mon + 0x2C)

            if is_double:
                if i < 2:
                    idx = p_count
                    self.p_att[idx] = attack
                    self.p_defn[idx] = defense
                    self.p_spe[idx] = speed
                    self.p_spA[idx] = spA
                    self.p_spD[idx] = spD
                    self.p_lvl[idx] = lvl
                    self.p_hp[idx] = hp / maxhp if maxhp else 0
                    self.p_exp[idx] = exp
                    self.p_ability[idx] = ability
                    self.p_moves[idx].copy_(torch.as_tensor(
                        moves,
                        dtype=torch.int64,
                        device=self.device
                    ))
                    self.p_item[idx] = holdItem
                    self.p_species[idx] = species
                    self.p_stat_changes[idx] = statChanges
                    self.p_pp[idx].copy_(torch.as_tensor(
                        pp,
                        dtype=torch.int64,
                        device=self.device
                    ))
                    self.p_types[idx].copy_(torch.as_tensor(
                        types,
                        dtype=torch.int64,
                        device=self.device
                    ))
                    self.p_status[idx] = status1
                    p_count += 1
                else:
                    idx = e_count
                    self.e_lvl[idx] = lvl
                    self.e_hp[idx] = hp / maxhp if maxhp else 0
                    e_count += 1
            else:
                if i == 0:
                    idx = 0
                    self.p_att[idx] = attack
                    self.p_defn[idx] = defense
                    self.p_spe[idx] = speed
                    self.p_spA[idx] = spA
                    self.p_spD[idx] = spD
                    self.p_lvl[idx] = lvl
                    self.p_hp[idx] = hp / maxhp if maxhp else 0
                    self.p_exp[idx] = exp
                    self.p_ability[idx] = ability
                    self.p_moves[idx].copy_(torch.as_tensor(
                        moves,
                        dtype=torch.int64,
                        device=self.device
                    ))
                    self.p_item[idx] = holdItem
                    self.p_species[idx] = species
                    self.p_stat_changes[idx] = statChanges
                    self.p_pp[idx].copy_(torch.as_tensor(
                        pp,
                        dtype=torch.int64,
                        device=self.device
                    ))
                    self.p_types[idx].copy_(torch.as_tensor(
                        types,
                        dtype=torch.int64,
                        device=self.device
                    ))
                    self.p_status[idx] = status1
                else:
                    idx = e_count
                    self.e_lvl[idx] = lvl
                    self.e_hp[idx] = hp / maxhp if maxhp else 0
                    e_count += 1

        self.double_battle = False

        pM = TensorDict({
            'att': self.p_att,
            'defn': self.p_defn,
            'spe': self.p_spe,
            'spA': self.p_spA,
            'spD': self.p_spD,
            'hp': self.p_hp,
            'lvl': self.p_lvl,
            'ability': self.p_ability,
            'lvl': self.p_lvl,
            'pp': self.p_pp,
            'exp': self.p_exp,
            'status1': self.p_status,
            'statChanges': self.p_stat_changes,
            'types': self.p_types,
            'species': self.p_species,
            'holdItem' : self.p_item,
            'moves' : self.p_moves,
        }, batch_size=[])

        eM = TensorDict({
            'hp': self.e_hp,
            'lvl': self.e_lvl,
            'status1': self.e_status,
        }, batch_size=[])

        return pM, eM
        
    
        





    
