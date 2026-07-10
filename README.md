# Pokemon EmeRLd

> **A Reinforcement Learning agent to beat the greatest pokemon game ever created.**

---

Would be hell of a lot more difficult without the decompilation of the game provided by pret @ https://github.com/pret/pokeemerald

---

## Extracting the RAM from Emulator

- For emulator I make use of BizHawk https://github.com/TASEmulators/BizHawk and have modified the built to expose API for extracting pixel data and RAM depending on the memory domain.

- Built a **C module** for Lua to store the extracted data from emulator into **shared memory** for fast IPC. [Serial](/mmap.c) for the serial implementation of the module and [Parallel](/par_mmap.c) when running multiple environments.

- The shm module makes use of semaphores for handling synchronization.

- Lua script [middleman](/middleman_par.lua) is used to access the ram and store the data into the shared memory. (For the serial implementation just remove the argument given to `create_shm` )


---

## Creating the Environment

 A custom environment using torchrl for the game which handles the shm initialization.

- [Memory](/Memory.py) class provides methods to access data in the ram depending on the size and type of data we want to access. (signed/unsigned 8,16,32 bit)

- [Pixel](/Memory.py) class provides access to the rgb values of each pixel in game.

- [MapNode](/Map.py) class used to manage each location visited in the game whether its a building/route/town/cave.

- [PokeCenter](/Map.py) class manages access to pokemon centers depending on the location of the player.

**The table below shows the relevant data extracted from the RAM with functions**

| Function | Description |
| ---- | ----------- |
| `get_textbox_flags` | Used to check whether interaction in overworld is happening and requires action with the help of status and active|
| `get_start_menu` | Used to get flags for start menu being opened and the current option selected  |
| `get_npcs` | Used to get the npcs available on the screen currently |
| `get_hms` | Used to get the hms the trainer currently holds |
| `get_coords` | Used to get current coordinates of player along with flags for direction, inside a PC, visited and a new map location |
| `get_party` | Maintains the players party and all the relevant information for each pokemon  |
| `get_badges` | Used to maintain number of badges the player holds |
| `get_battle_info` | Used to maintain the information for player and enemy pokemon during battle |

## Working of Environment

The environment maintains the state of the game for each frame with the help of **semaphores**.

---
### Reward Calculation (Current)

- Takes *current and previous state* (previous = last frame) as arguments.

- Reward of - 0.001 for every step of environment.

- Story flags for intro of the game with + 10 reward with rest of the rewards locked till clock is set in game.

- Keeps track of location of player to manage **stuck** flag.

- Maximum reward of + 100 for obtaining a gym badge.

- Movement reward + 0.1 for every new tile visited with the a decay function of $e^{-0.99 * count of visits}$ and a reward of + 20 for every new location visited on map with the same decay function.

- HM reward of + 50 for obtaining a HM.

- PC reward of +0.5 if entering a PC when party is low on HP and + 1.5 reward for healing at the PC.

- NPC interaction reward of + 1 .

- Post battle rewards calculated based on outcome of battle, for loss - 2, for win + 1 mulitplied by the same reward decay function used for movement but with count of number of battles used and + 0.5 * $reward decay$ when party health is <=30% .

- In battle rewards maintained to be added in post-battle depending on hp,status and exp change.

---

## [Encoders](/Encoders.py)
---
### Player Encoder

> **Creates embeddings for pokemons on players side during battle making use of MLP.**

Pokemon on player side contains the following features :
- Species
- Move
- Ability
- Status
- Item held
- Hp
- Level
- Attack
- Defence
- Speed
- Special Attack
- Special Defence
- PP
- Experience

The first 5 features are fed into MLP to create embeddings as they convey categorical information.
Size = 128

---
### Enemy Encoder

> **Creates embedding for pokemons on enemy side during battle making use of MLP.**

Pokemon on enemy side contains the following features :
- Status
- Hp
- Level

Here only status is converted into an embedding as it conveys categorical information.
Size = 128

---
### PixelEncoder

> **A CNN designed to convert the pixel data obtained from the emulator into latent embeddings to be used by Policy head.**

Takes in the rgb values obtained from the pixel data of emulator.
Size = 512

---
### StateEncoder

> **Contains a Transformer for converting battle information into embeddings along with combining embeddings obtained from other encoders into final latent vector to be given to Policy.**

First the embeddings are obtained from player and enemy encoders then projected into tokens with 256 dimensions along with a embedding for which side the token belongs to and concanated into a single vector which is reshaped to match the transformer and outputs the final battle representation.

Combines all other embeddings with battle representation which is then passed to a final MLP to form the final vector that is going to be fed to the policy network.




## [Policy](/Policy.py)

> **Contains definition for the Actor and Critic network.**

### Actor

> **Obtains the final vector from State Encoder and then feed it into a neural network.**

The Actor has 2 neural networks inside it, one for **overworld** other for **battle**. This division was done to ensure a better policy as the 2 scenarios are completely different from each other and contain different rewards.

### Critic

> **Makes use of State Encoder and calculates the value function.**

Has a 3 layer MLP for calculating the value function.





## Trainer

> **Used to handle the training of RL model**

The trainer is used to train the agent. It contains all the hyperparameters along with logging for generating values of relevant metrics after each batch is processed.

- Uses ParallelEnv provided by PyTorch for paralleling running multiple instances of environment.
- Uses ProbabilisticActor for getting the action.
- Uses ValueOperator for generating value function.
- Uses SyncDataCollector for managing the environments.
- Uses PPO algorithm for training.





