import json 

class MapNode :

    def __init__(self,id):

        self.mapId = id
        self.visited = {}

    def update_visit(self, x, y):
        if y in self.visited.get(x, []):
            return 1

        self.visited.setdefault(x, []).append(y)
        return 0
    

class PokeMonCenter :

    def __init__(self,mapgroupfile) :

        self.pcs = self._init_poke_centers(mapgroupfile)

    def _init_poke_centers(self,file):

        with open(file,'r') as f :
            data = json.load(f)
        pokemon_centers = {}

        towns = ['Oldale',
                 'Dewford',
                 'Lavaridge',
                 'Fallarbor',
                 'Verdanturf',
                 'Pacifidlog',
                 'Petalburg',
                 'Slateport',
                 'Mauville',
                 'Rustboro',
                 'Fortree',
                 'Lilycove',
                 'Mossdeep',
                 'Sootopolis',
                 'EverGrande'
                ]
        
        towns_group_number = {"gMapGroup_IndoorOldale" : 1,
                        "gMapGroup_IndoorDewford" : 2,
                        "gMapGroup_IndoorLavaridge" : 3,
                        "gMapGroup_IndoorFallarbor" : 4,
                        "gMapGroup_IndoorVerdanturf" : 5,
                        "gMapGroup_IndoorPacifidlog" : 6,
                        "gMapGroup_IndoorPetalburg" : 7,
                        "gMapGroup_IndoorSlateport" : 8,
                        "gMapGroup_IndoorMauville" : 9,
                        "gMapGroup_IndoorRustboro" : 10,
                        "gMapGroup_IndoorFortree" : 11,
                        "gMapGroup_IndoorLilycove" : 12,
                        "gMapGroup_IndoorMossdeep" : 13,
                        "gMapGroup_IndoorSootopolis" : 14,
                        "gMapGroup_IndoorEverGrande" : 15
                    }

        for key,value in data.items():
            if key in towns_group_number.keys() :
                map_grp = towns_group_number[key]
                map_num = 0
                for i,place in enumerate(value) :
                    if 'PokemonCenter_1F' in place :
                        map_num = i
                        break
                pokemon_centers[towns[map_grp - 1]] = ((map_grp + 1) << 8) + map_num

        return pokemon_centers



    
        
        
