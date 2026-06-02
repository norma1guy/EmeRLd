import traceback

import posix_ipc,mmap,struct,torch
from Memory import Memory, Pixels
from Map import MapNode,PokeMonCenter
from tensordict import TensorDict
from torchrl.data import DiscreteTensorSpec,BoundedTensorSpec, CompositeSpec, UnboundedDiscreteTensorSpec,UnboundedContinuousTensorSpec
from torchrl.envs import EnvBase
from typing import Optional
import numpy as np
from Battle import BattlePokemon

class Environment(EnvBase) :

    def __init__(self,td_params=None,seed=None,device='cuda' if torch.cuda.is_available() else 'cpu'):

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
        self.iwru8 = self.iwram.read_u8
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
        self.coords = torch.zeros(7, device=self.device, dtype=torch.int64)

        self.badges = torch.zeros(8, device=self.device, dtype=torch.int64)
        self.npcs_seen = {}
        self.player_battle_mon = BattlePokemon()
        self.enemy_battle_mon = BattlePokemon()
        self.out_td = TensorDict({}, batch_size=[])
        self.battle_rewards = 0
        self.battle_turns = 0
        self.turn_active = 0
        self.battles = 0
        self.poke_centers = PokeMonCenter('map_groups.json')
        self.steps = 0
        self.whiteouts = 0

    def _get_state(self):
        

        # In battle from gMain + 0x439 3rd bit
        in_battle_byte = self.iwru8(0x026f9) // 2
        self.is_battle = in_battle_byte % 2
        battle_type = self.ru32le(0x22fec)
        mapState = self.get_coords()
        active,state = self._get_textbox_flags()
        badges = self.get_badges()
        party = self.get_party()
        hms = self.get_hms()
        player_battle_state,enemy_battle_state = self.get_battle_info()
        battleoutcome = self.ru8(0x2433a)

        out = TensorDict(
            {
                'inbattle' : torch.tensor([self.is_battle],dtype=torch.int64,device=self.device),
                'battleoutcome' : torch.tensor([battleoutcome],dtype=torch.int64,device=self.device),
                'battletype' : torch.tensor([battle_type],dtype=torch.int64,device=self.device),
                'textactive' : torch.tensor([active],dtype=torch.int64,device=self.device),
                'textstate' : torch.tensor([state],dtype=torch.int64,device=self.device),
                'playerpokemon' : player_battle_state,
                'enemypokemon' : enemy_battle_state,
                'map' : mapState,
                'badge' : badges,
                'party' : party,
                'hms' : hms,
            },
            batch_size=[]
        )
        
        return out.to(device=self.device)
    
    def _make_spec(self):
        

        self.observation_spec = CompositeSpec(

            observation = CompositeSpec(
                inbattle = DiscreteTensorSpec(2, shape=(1,), dtype=torch.int64),
                battleoutcome = DiscreteTensorSpec(5, shape=(1,), dtype=torch.int64),
                battletype = UnboundedDiscreteTensorSpec(shape=(1,),dtype=torch.int64),
                textactive = DiscreteTensorSpec(2, shape=(1,), dtype=torch.int64),
                textstate = UnboundedDiscreteTensorSpec(shape=(1,),dtype=torch.int64),
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
                map = UnboundedDiscreteTensorSpec(shape=(7,),dtype=torch.int64),
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
            ),
            steps = UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            )
        )

        self.action_spec = DiscreteTensorSpec(
            n=7,
            shape=(1,),
            dtype=torch.int64,
        )

        self.reward_spec = UnboundedContinuousTensorSpec(
            shape=(1,),
            dtype=torch.float32,
        )

        self.done_spec = CompositeSpec(
            done=DiscreteTensorSpec(
                n=2,
                shape=(1,),
                dtype=torch.bool,
            ),
            terminated=DiscreteTensorSpec(
                n=2,
                shape=(1,),
                dtype=torch.bool,
            ),
        )



    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def calc_reward(self,prev_state,next_state) :
        reward = 0
        party_count = self.ru8(0x244e9)
        #print(next_state['map'])
        # Badge reward
        if prev_state is None :
            pass
        elif next_state['badge'].sum() > prev_state['badge'].sum() :
            reward += 10 
        
        # Movement reward
        if not torch.equal(next_state['map'],prev_state['map']) :
            visited_flag = next_state['map'][5].item()
            new_place = next_state['map'][6].item()
            if not visited_flag and new_place :
                reward += 1
                #print(69)
        #print(next_state['inbattle'])      

        # HMs reward
        if prev_state is None :
            pass
        elif next_state['hms'].sum() > prev_state['hms'].sum() :
            reward += 10

        # PokeCenter reward
        if not next_state['inbattle'].item() :
            if next_state['map'][3].item() == 1 and next_state['party']['hp'].sum() / party_count < 0.3:
                reward += 0.5 
            for i,j in zip(next_state['party']['hp'],prev_state['party']['hp']) :
                if i > j and j.item() <= 0.3:
                    reward += 1.5
    
        # NPC interaction reward
        active,status = self._get_textbox_flags()
        if active :
            npcs = self._get_npcs()

            for key,npc in npcs.items() :
                diff1 = abs(next_state['map'][0].item() - npc[0])
                diff2 = abs(next_state['map'][1].item() - npc[1])
                if ((diff1 == 0 and diff2 == 1) or (diff2 == 0 and diff1 == 1)) and self.npcs_seen[key] == 0 :
                    reward += 0.1
                    self.npcs_seen[key] += 1
                    break

        
        #print(self.battle_rewards)
        #Post-Battle reward
        if not next_state['inbattle'].item() and prev_state['inbattle'].item():
            #print(self.battle_turns)
            next_hp = next_state['party']['hp'].sum()
            prev_lvl = prev_state['party']['level'].sum()
            next_lvl = next_state['party']['level'].sum()
            next_status = prev_state['party']['status']
            battle_outcome = next_state['battleoutcome']
            #self.battles += 1

            # Status penalty
            if (next_status != 0).any():
                reward -= 0.2

            # Level reward
            if next_lvl > prev_lvl:
                reward += 1

            # Outcome reward
            if battle_outcome == 1 :
                reward += self.battle_rewards
            if battle_outcome == 2:
                reward -= 1
                self.whiteouts += 1

            if battle_outcome == 4 and next_hp / party_count <= 0.3:
                reward += min(0.3 * self.battle_rewards * (1 - pow(0.9,self.battle_turns)) - 5,1) / self.battles if self.battles else min(0.3 * self.battle_rewards * (1 - pow(0.9,self.battle_turns)) - 5,1)
            elif battle_outcome == 4 and next_hp / party_count > 0.3 :
                reward -= 0.5
            self.battle_rewards = 0
            self.battle_turns = 0
            self.player_battle_mon.reset()
            self.enemy_battle_mon.reset()
            #print('Post Battle Reward ',reward)

        # Battle initialization
        elif next_state['inbattle'].item() and not prev_state['inbattle'].item() :

            #Player
            self.player_battle_mon.update(
                hp=next_state['playerpokemon']['hp'].sum().item(),
                exp=next_state['playerpokemon']['exp'].sum().item(),
                status=next_state['playerpokemon']['status1'].sum().item()
            )

            #Enemy
            self.enemy_battle_mon.update(
                hp=next_state['enemypokemon']['hp'].sum().item(),
                status=next_state['enemypokemon']['status1'].sum().item()
            )
            reward += 0.1
            self.battles += 1
            #print(self.player_battle_mon.hp,self.enemy_battle_mon.hp)
            
        # In Battle Rewards        
        elif next_state['inbattle'].item() and prev_state['inbattle'].item():
            turn_action_number = self.ru8(0x24082)
            if turn_action_number != 2 :
                self.turn_active = 1

            elif turn_action_number  == 2 and self.turn_active :
                self.battle_turns += 1
                self.turn_active = 0

            #Enemy
            enemy_next_hp = next_state['enemypokemon']['hp'].sum().item()
            enemy_next_status = next_state['enemypokemon']['status1'].sum().item()

            #Player
            player_next_hp = next_state['playerpokemon']['hp'].sum().item()
            player_next_exp = next_state['playerpokemon']['exp'].sum().item()

            #print('ENEMY:',enemy_next_hp,self.enemy_battle_mon.hp)
            #print('PLAYER:',player_next_hp,self.player_battle_mon.hp)

            if enemy_next_hp < self.enemy_battle_mon.hp :
                hp_taken = (self.enemy_battle_mon.hp - enemy_next_hp)
                self.battle_rewards  += 0.25 * hp_taken
                self.enemy_battle_mon.update(hp=enemy_next_hp)

            if self.enemy_battle_mon.status == 0 and enemy_next_status != 0 :
                self.battle_rewards += 0.5
                self.enemy_battle_mon.update(status=enemy_next_status)

            if player_next_hp < self.player_battle_mon.hp :
                hp_lost = (self.player_battle_mon.hp - player_next_hp)
                self.battle_rewards -= 0.25 * hp_lost
                self.player_battle_mon.update(hp=player_next_hp)
            
            if player_next_exp > self.player_battle_mon.hp :
                exp_gained = (player_next_exp - self.player_battle_mon.exp)
                self.battle_rewards += 0.05 * min(5,exp_gained)
                self.player_battle_mon.update(exp=player_next_exp)
            #print('BR: ',self.battle_rewards)
        if reward > 10 :
            reward = 10
        return torch.tensor(reward, dtype=torch.float32,device=self.device)
    
    def _step(self, tensordict):

        
        action = tensordict.get('action')
        act_val = int(action)  

        prev_obs = tensordict['observation']  
        self.luaFlag.acquire()
        self.mm[:4] = struct.pack("I", act_val)
        next_obs = self._get_state()
        reward = self.calc_reward(prev_obs, next_obs)

        self.pyFlag.release()

        # Defeating all gyms ends the episode
        done = next_obs['badge'].sum() == 8
        terminated = torch.tensor(
            [True if self.whiteouts == 5 else False],
            dtype=torch.bool,
            device=self.device
        )
        reward = reward.unsqueeze(0)

        done = torch.tensor(
            [done],
            dtype=torch.bool,
            device=self.device
        )
        self.steps += 1

        return TensorDict(
            {
                "observation": next_obs,
                "reward": reward,
                "done": done,
                "terminated" : terminated,
                "steps" : torch.tensor([self.steps],dtype=torch.int64,device=self.device)
            },
            batch_size=[]
        )
    
    def _reset(self, tensordict):

        self.luaFlag.acquire()

        self.mm[:4] = struct.pack('I', 7)

        self.pyFlag.release()

        if tensordict is None:
            tensordict = TensorDict({}, batch_size=[])

        tensordict['observation'] = self._get_state()
        tensordict["done"] = torch.tensor(
            [False],
            dtype=torch.bool,
            device=self.device
        )

        tensordict["terminated"] = torch.tensor(
            [False],
            dtype=torch.bool,
            device=self.device
        )

        tensordict["steps"] = torch.tensor(
            [0],
            dtype=torch.int64,
            device=self.device
        )

        self.battles = 0
        self.battle_rewards = 0
        self.turn_active = 0
        self.battle_turns = 0
        self.steps = 0
        self.whiteouts = 0
        #print("RESET CALLED")

        return tensordict
            
    def _get_textbox_flags(self) :

        active = self.ru8(0x0201cb)
        state = self.ru8(0x0201cc)

        return (active,state)

    def _get_npcs(self) :
        gObjectEvents = 0x37350
        npcs = {}
        for i in range(16) :
            start = gObjectEvents + i * 36
            info = self.ru8(start + 0x02)
            dir = self.ru8(start + 24)
            #print(start)
            is_player = (info >> (info.bit_length() - 1) & 1) if info else 0
            current_coords = [self.rs16le(start + 16 + i * 2) for i in range(2)]
            #sprite_id = self.ru8(start + 4)
            #graphics_id = self.ru8(start + 5)
            #mov_type = self.ru8(start + 6)
            #trainer_type = self.ru8(start + 7)
            local_id = self.ru8(start + 8)
            map_num = self.ru8(start + 9)
            map_group = self.ru8(start + 10)
            if is_player :
                key = (local_id,map_num,map_group)
                npcs[key] = current_coords
                if key not in self.npcs_seen.keys() :
                    self.npcs_seen[key] = 0
            
            
            
            '''print({
                'is_player' : is_player,
                'direction' : dir,
                'sprite' : sprite_id,
                'grapics' : graphics_id,
                'mov_type' : mov_type,
                'trainer_type' : trainer_type,
                'coords' : current_coords
            })'''
        return npcs
        #print('FUUUUUUUUUUUUCCCCCCCCCCCCCCKKKKKKKKKKKKKKKKKKKKK')
    
    def get_hms(self):
        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000
        base = saveBlockAddr + 0x690 + 339

        for i in range(8):
            self.hms[i] = 1 if self.ru16le(base + 4 * i) else 0

        return self.hms.clone()

   
    def get_coords(self):

        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000

        x = self.rs16le(saveBlockAddr)
        y = self.rs16le(saveBlockAddr + 0x2)
        direction = self.rs16le(0x037368) // 17
        mapId = self.ru16le(saveBlockAddr + 0x32)
        map_group = self.rs8(saveBlockAddr + 0x04)
        map_num = self.rs8(saveBlockAddr + 0x05)
        new_place = None
        is_poke_center = 1 if (map_group >> 8) + map_num in self.poke_centers.pcs.values() else 0
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
        self.coords[3] = is_poke_center
        self.coords[4] = mapId
        self.coords[5] = visited
        self.coords[6] = new_place

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
        # Flags are stored as bits
        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000
        flags_start = saveBlockAddr + 0x1270
        gym_flag = 0x868
        gym_2_to_8 =  self.ru8(flags_start + gym_flag // 8)

        for i in range(7,0,-1) :
            self.badges[i] = gym_2_to_8 % 2
            gym_2_to_8 = gym_2_to_8 // 2

        gym1 = self.ru8(flags_start + gym_flag // 8 - 1)
        for i in range(7):
            gym1 = gym1 // 2
        self.badges[0] = gym1

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
        if self.is_battle :
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
            'pp': self.p_pp,
            'exp': self.p_exp,
            'status1': self.p_status,
            'statChanges': self.p_stat_changes,
            'types': self.p_types,
            'species': self.p_species,
            'holdItem' : self.p_item,
            'moves' : self.p_moves,
        }, batch_size=[],device=self.device)

        eM = TensorDict({
            'hp': self.e_hp,
            'lvl': self.e_lvl,
            'status1': self.e_status,
        }, batch_size=[],device=self.device)

        return pM, eM
        
    


class ParallelEnvironment(EnvBase) :

    def __init__(self,num_proc,seed=None,device='cuda'):

        super().__init__(device=device,batch_size=[])
        self._make_spec()
        if seed is None:
            seed = torch.empty((), dtype=torch.int64).random_().item()
        self.set_seed(seed)
        #print(num_proc)
        self.shm = posix_ipc.SharedMemory(f"/RAM_MAP{num_proc}")
        self.pyFlag = posix_ipc.Semaphore(f"/py_to_lua{num_proc}")
        self.luaFlag  = posix_ipc.Semaphore(f"/lua_to_py{num_proc}")
        self.ewramSize = 256 * 1024
        self.iwramSize = 32 * 1024
        self.pixelCount = 240 * 160
        self.pixelSize = self.pixelCount * 4
        self.mm = mmap.mmap(self.shm.fd,8 + self.ewramSize + self.iwramSize + self.pixelSize)
        self.shm.close_fd()
        self.input = struct.unpack("I",self.mm[:4])
        self.ewram = Memory(self.mm,8)
        self.iwram = Memory(self.mm,8 + self.ewramSize)
        self.pixels = Pixels(self.mm,self.pixelCount,8 + self.ewramSize + self.iwramSize)
        self.is_battle = False
        self.max_battlers = 4
        self.double_battle = False
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
        self.iwru8 = self.iwram.read_u8
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
        self.coords = torch.zeros(7, device=self.device, dtype=torch.int64)

        self.badges = torch.zeros(8, device=self.device, dtype=torch.int64)
        self.npcs_seen = {}
        self.player_battle_mon = BattlePokemon()
        self.enemy_battle_mon = BattlePokemon()
        self.out_td = TensorDict({}, batch_size=[])
        self.battle_rewards = 0
        self.battle_turns = 0
        self.turn_active = 0
        self.battles = 0
        self.poke_centers = PokeMonCenter('map_groups.json')
        self.steps = 0
        self.whiteouts = 0
        self.stucksteps = 0


    def _get_state(self):
        

        # In battle from gMain + 0x439 3rd bit
        in_battle_byte = self.iwru8(0x026f9) // 2
        self.is_battle = in_battle_byte % 2
        battle_type = self.ru32le(0x22fec)
        mapState = self.get_coords()
        active,state = self._get_textbox_flags()
        badges = self.get_badges()
        party = self.get_party()
        hms = self.get_hms()
        player_battle_state,enemy_battle_state = self.get_battle_info()
        battleoutcome = self.ru8(0x2433a)
        menu_flags = self._get_start_menu()
        pixels = torch.from_numpy(self.pixels.rgb).float() / 255.0
        pixels = pixels.permute(2,0,1).unsqueeze(0)

        out = TensorDict(
            {
                'inbattle' : torch.tensor([self.is_battle],dtype=torch.int64,device=self.device),
                'battleoutcome' : torch.tensor([battleoutcome],dtype=torch.int64,device=self.device),
                'battletype' : torch.tensor([battle_type],dtype=torch.int64,device=self.device),
                'textactive' : torch.tensor([active],dtype=torch.int64,device=self.device),
                'textstate' : torch.tensor([state],dtype=torch.int64,device=self.device),
                'playerpokemon' : player_battle_state,
                'enemypokemon' : enemy_battle_state,
                'map' : mapState,
                'badge' : badges,
                'party' : party,
                'hms' : hms,
                'startmenu' : menu_flags,
                'pixelbuffer' : pixels,
            },
            batch_size=[]
        )
        
        return out.to(device=self.device)
    
    def _make_spec(self):
        

        self.observation_spec = CompositeSpec(

            observation = CompositeSpec(
                inbattle = DiscreteTensorSpec(2, shape=(1,), dtype=torch.int64),
                battleoutcome = DiscreteTensorSpec(5, shape=(1,), dtype=torch.int64),
                battletype = UnboundedDiscreteTensorSpec(shape=(1,),dtype=torch.int64),
                textactive = DiscreteTensorSpec(2, shape=(1,), dtype=torch.int64),
                textstate = UnboundedDiscreteTensorSpec(shape=(1,),dtype=torch.int64),
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
                map = UnboundedDiscreteTensorSpec(shape=(7,),dtype=torch.int64),
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
                startmenu = UnboundedDiscreteTensorSpec(
                    shape=(2,),
                    dtype=torch.int64,
                ),
                pixelbuffer = BoundedTensorSpec(
                    shape=(1,3,160,240),
                    low=0.00,
                    high=1.0,
                    dtype=torch.float32,
                )
            ),
            envsteps = UnboundedDiscreteTensorSpec(
                shape=(1,),
                dtype=torch.int64,
            )
        )

        self.action_spec = DiscreteTensorSpec(
            n=7,
            shape=(),
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

    def calc_reward(self,prev_state,next_state) :
        reward = 0
        party_count = self.ru8(0x244e9)
        
        # Stuck Counter
        current_location = (next_state['map'][0].item(),next_state['map'][1].item(),next_state['map'][4].item())
        previous_location = (prev_state['map'][0].item(),prev_state['map'][1].item(),prev_state['map'][4].item())
        if current_location == previous_location :
            self.stucksteps += 1
        else :
            self.stucksteps = 0 

        # Badge reward
        if prev_state is None :
            pass
        elif next_state['badge'].sum() > prev_state['badge'].sum() :
            reward += 10 
        
        # Movement reward
        if not torch.equal(next_state['map'],prev_state['map']) :
            visited_flag = next_state['map'][5].item()
            new_place = next_state['map'][6].item()
            if not visited_flag and new_place :
                reward += 1
                #print(69)
        #print(next_state['inbattle'])      

        # HMs reward
        if prev_state is None :
            pass
        elif next_state['hms'].sum() > prev_state['hms'].sum() :
            reward += 10

        # PokeCenter reward
        if not next_state['inbattle'] :
            if next_state['map'][3].item() == 1 and next_state['party']['hp'].sum() / party_count < 0.3:
                reward += 0.5 
            for i,j in zip(next_state['party']['hp'],prev_state['party']['hp']) :
                if i > j and j.item() <= 0.3:
                    reward += 1.5
    
        # NPC interaction reward
        active,status = self._get_textbox_flags()
        if active :
            npcs = self._get_npcs()

            for key,npc in npcs.items() :
                diff1 = abs(next_state['map'][0].item() - npc[0])
                diff2 = abs(next_state['map'][1].item() - npc[1])
                if ((diff1 == 0 and diff2 == 1) or (diff2 == 0 and diff1 == 1)) and self.npcs_seen[key] == 0 :
                    reward += 0.1
                    self.npcs_seen[key] += 1
                    break
        
        # Start Menu
        if next_state['startmenu'][0].item() != 255 and next_state['startmenu'][1].item() > 3:
            reward -= 0.05

        
        #Post-Battle reward
        if not next_state['inbattle'].item() and prev_state['inbattle'].item():
            next_hp = next_state['party']['hp'].sum()
            prev_lvl = prev_state['party']['level'].sum()
            next_lvl = next_state['party']['level'].sum()
            next_status = prev_state['party']['status']
            battle_outcome = next_state['battleoutcome']

            # Status penalty
            if (next_status != 0).any():
                reward -= 0.2

            # Level reward
            if next_lvl > prev_lvl:
                reward += 1

            # Outcome reward
            if battle_outcome == 1 :
                reward += self.battle_rewards
            if battle_outcome == 2:
                reward -= 1
                self.whiteouts += 1

            if battle_outcome == 4 and next_hp / party_count <= 0.3:
                reward += min(0.3 * self.battle_rewards * (1 - pow(0.9,self.battle_turns)) - 5,1) / self.battles if self.battles else min(0.3 * self.battle_rewards * (1 - pow(0.9,self.battle_turns)) - 5,1)
            elif battle_outcome == 4 and next_hp / party_count > 0.3 :
                reward -= 0.5
            self.battle_rewards = 0
            self.battle_turns = 0
            self.player_battle_mon.reset()
            self.enemy_battle_mon.reset()
            #print('Post Battle Reward ',reward)

        # Battle initialization
        elif next_state['inbattle'].item() and not prev_state['inbattle'].item() :

            #Player
            self.player_battle_mon.update(
                hp=next_state['playerpokemon']['hp'].sum().item(),
                exp=next_state['playerpokemon']['exp'].sum().item(),
                status=next_state['playerpokemon']['status1'].sum().item()
            )

            #Enemy
            self.enemy_battle_mon.update(
                hp=next_state['enemypokemon']['hp'].sum().item(),
                status=next_state['enemypokemon']['status1'].sum().item()
            )
            reward += 0.1
            self.battles += 1
            #print(self.player_battle_mon.hp,self.enemy_battle_mon.hp)
            
        # In Battle Rewards        
        elif next_state['inbattle'].item() and prev_state['inbattle'].item():
            turn_action_number = self.ru8(0x24082)
            if turn_action_number != 2 :
                self.turn_active = 1

            elif turn_action_number  == 2 and self.turn_active :
                self.battle_turns += 1
                self.turn_active = 0

            #Enemy
            enemy_next_hp = next_state['enemypokemon']['hp'].sum().item()
            enemy_next_status = next_state['enemypokemon']['status1'].sum().item()

            #Player
            player_next_hp = next_state['playerpokemon']['hp'].sum().item()
            player_next_exp = next_state['playerpokemon']['exp'].sum().item()

            #print('ENEMY:',enemy_next_hp,self.enemy_battle_mon.hp)
            #print('PLAYER:',player_next_hp,self.player_battle_mon.hp)

            if enemy_next_hp < self.enemy_battle_mon.hp :
                hp_taken = (self.enemy_battle_mon.hp - enemy_next_hp)
                self.battle_rewards  += 0.25 * hp_taken
                self.enemy_battle_mon.update(hp=enemy_next_hp)

            if self.enemy_battle_mon.status == 0 and enemy_next_status != 0 :
                self.battle_rewards += 0.5
                self.enemy_battle_mon.update(status=enemy_next_status)

            if player_next_hp < self.player_battle_mon.hp :
                hp_lost = (self.player_battle_mon.hp - player_next_hp)
                self.battle_rewards -= 0.25 * hp_lost
                self.player_battle_mon.update(hp=player_next_hp)
            
            if player_next_exp > self.player_battle_mon.hp :
                exp_gained = (player_next_exp - self.player_battle_mon.exp)
                self.battle_rewards += 0.05 * min(5,exp_gained)
                self.player_battle_mon.update(exp=player_next_exp)
            
        '''if reward > 10 :
            reward = 10'''
        return [torch.tensor(reward, dtype=torch.float32,device=self.device),self.stucksteps >= 500]
    
    def _step(self, tensordict):

        
        action = tensordict.get('action')
        #print("ACTION SHAPE:", action.shape)
        act_val = int(action)  
        #print(action.shape)
        prev_obs = tensordict['observation']  
        self.luaFlag.acquire()
        self.mm[:4] = struct.pack("I", act_val)
        next_obs = self._get_state()
        reward,stuck_flag = self.calc_reward(prev_obs, next_obs)
        #print('Stuck',self.stucksteps,stuck_flag)
        self.pyFlag.release()

        # Defeating all gyms ends the episode
        done = next_obs['badge'].sum() == 8
        terminated = self.whiteouts == 5 or stuck_flag
        done = done or terminated
        reward = reward.unsqueeze(0)

        done = torch.tensor(
            [done],
            dtype=torch.bool,
            device=self.device
        )
        self.steps += 1
        td = TensorDict(
            {
                "observation": next_obs,
                "reward": reward,
                "done": done,
                "envsteps" : torch.tensor([self.steps],dtype=torch.int64,device=self.device)
            },
            batch_size=[]
        )

        return td
    
    def _reset(self, tensordict):

        self.luaFlag.acquire()

        self.mm[:4] = struct.pack('I', 7)

        self.pyFlag.release()

        if tensordict is None:
            tensordict = TensorDict({}, batch_size=[])

        tensordict['observation'] = self._get_state()
        tensordict["done"] = torch.tensor(
            [False],
            dtype=torch.bool,
            device=self.device
        )

        tensordict["envsteps"] = torch.tensor(
            [0],
            dtype=torch.int64,
            device=self.device
        )

        self.battles = 0
        self.battle_rewards = 0
        self.turn_active = 0
        self.battle_turns = 0
        self.steps = 0
        self.whiteouts = 0
        self.stucksteps = 0
        #print("RESET CALLED")
        #print(tensordict['terminated'].item(),tensordict['done'].item(),tensordict['envsteps'].item())
        
        return tensordict
            
    def _get_textbox_flags(self) :

        active = self.ru8(0x0201cb)
        state = self.ru8(0x0201cc)

        return (active,state)
    
    def _get_start_menu(self) :

        start_menu_flag = 0x3cd8c
        menu_cursor = 0x3760e

        return [self.ru8(start_menu_flag),self.ru8(menu_cursor)]

    def _get_npcs(self) :
        gObjectEvents = 0x37350
        npcs = {}
        for i in range(16) :
            start = gObjectEvents + i * 36
            info = self.ru8(start + 0x02)
            dir = self.ru8(start + 24)
            #print(start)
            is_player = (info >> (info.bit_length() - 1) & 1) if info else 0
            current_coords = [self.rs16le(start + 16 + i * 2) for i in range(2)]
            #sprite_id = self.ru8(start + 4)
            #graphics_id = self.ru8(start + 5)
            #mov_type = self.ru8(start + 6)
            #trainer_type = self.ru8(start + 7)
            local_id = self.ru8(start + 8)
            map_num = self.ru8(start + 9)
            map_group = self.ru8(start + 10)
            if is_player :
                key = (local_id,map_num,map_group)
                npcs[key] = current_coords
                if key not in self.npcs_seen.keys() :
                    self.npcs_seen[key] = 0
            
            
            
            '''print({
                'is_player' : is_player,
                'direction' : dir,
                'sprite' : sprite_id,
                'grapics' : graphics_id,
                'mov_type' : mov_type,
                'trainer_type' : trainer_type,
                'coords' : current_coords
            })'''
        return npcs
        #print('FUUUUUUUUUUUUCCCCCCCCCCCCCCKKKKKKKKKKKKKKKKKKKKK')
    
    def get_hms(self):
        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000
        base = saveBlockAddr + 0x690 + 339

        for i in range(8):
            self.hms[i] = 1 if self.ru16le(base + 4 * i) else 0

        return self.hms.clone()

   
    def get_coords(self):

        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000

        x = self.rs16le(saveBlockAddr)
        y = self.rs16le(saveBlockAddr + 0x2)
        direction = self.rs16le(0x037368) // 17
        mapId = self.ru16le(saveBlockAddr + 0x32)
        map_group = self.rs8(saveBlockAddr + 0x04)
        map_num = self.rs8(saveBlockAddr + 0x05)
        new_place = None
        is_poke_center = 1 if (map_group >> 8) + map_num in self.poke_centers.pcs.values() else 0
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
        self.coords[3] = is_poke_center
        self.coords[4] = mapId
        self.coords[5] = visited
        self.coords[6] = new_place

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
            'status': self.party_status.clone(),
            'hp': self.party_hp.clone(),
            'level': self.party_level.clone(),
            },
            batch_size=[]
        )
        return out.to(device=self.device)
    
    def get_badges(self):
        # Flags are stored as bits
        saveBlockAddr = self.iwru32le(0x5d8c) - 0x2000000
        flags_start = saveBlockAddr + 0x1270
        gym_flag = 0x868
        gym_2_to_8 =  self.ru8(flags_start + gym_flag // 8)

        for i in range(7,0,-1) :
            self.badges[i] = gym_2_to_8 % 2
            gym_2_to_8 = gym_2_to_8 // 2

        gym1 = self.ru8(flags_start + gym_flag // 8 - 1)
        for i in range(7):
            gym1 = gym1 // 2
        self.badges[0] = gym1

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
        if self.is_battle :
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
            'att': self.p_att.clone(),
            'defn': self.p_defn.clone(),
            'spe': self.p_spe.clone(),
            'spA': self.p_spA.clone(),
            'spD': self.p_spD.clone(),
            'hp': self.p_hp.clone(),
            'lvl': self.p_lvl.clone(),
            'ability': self.p_ability.clone(),
            'pp': self.p_pp.clone(),
            'exp': self.p_exp.clone(),
            'status1': self.p_status.clone(),
            'statChanges': self.p_stat_changes.clone(),
            'types': self.p_types.clone(),
            'species': self.p_species.clone(),
            'holdItem' : self.p_item.clone(),
            'moves' : self.p_moves.clone(),
        }, batch_size=[],device=self.device)

        eM = TensorDict({
            'hp': self.e_hp.clone(),
            'lvl': self.e_lvl.clone(),
            'status1': self.e_status.clone(),
        }, batch_size=[],device=self.device)

        return pM, eM
        
    
        





    
