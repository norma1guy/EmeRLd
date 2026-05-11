from collections import defaultdict
from dataclasses import dataclass

class Map : 
    def __init__(self) :
        self.R = {
            "NONE": 0,
            "ROUTE_101": 1,
            "ROUTE_102": 2,
            "ROUTE_103": 3,
            "ROUTE_104": 4,
            "ROUTE_105": 5,
            "ROUTE_106": 6,
            "ROUTE_107": 7,
            "ROUTE_108": 8,
            "ROUTE_109": 9,
            "ROUTE_110": 10,
            "ROUTE_111": 11,
            "ROUTE_112": 12,
            "ROUTE_113": 13,
            "ROUTE_114": 14,
            "ROUTE_115": 15,
            "ROUTE_116": 16,
            "ROUTE_117": 17,
            "ROUTE_118": 18,
            "ROUTE_119": 19,
            "ROUTE_120": 20,
            "ROUTE_121": 21,
            "ROUTE_122": 22,
            "ROUTE_123": 23,
            "ROUTE_124": 24,
            "ROUTE_125": 25,
            "ROUTE_126": 26,
            "ROUTE_127": 27,
            "ROUTE_128": 28,
            "ROUTE_129": 29,
            "ROUTE_130": 30,
            "ROUTE_131": 31,
            "ROUTE_132": 32,
            "ROUTE_133": 33,
            "ROUTE_134": 34,

            "LITTLEROOT_TOWN": 40,
            "OLDALE_TOWN": 41,
            "PETALBURG_CITY": 42,
            "RUSTBORO_CITY": 43,
            "DEWFORD_TOWN": 44,
            "SLATEPORT_CITY": 45,
            "MAUVILLE_CITY": 46,
            "VERDANTURF_TOWN": 47,
            "LAVARIDGE_TOWN": 48,
            "FALLARBOR_TOWN": 49,
            "FORTREE_CITY": 50,
            "LILYCOVE_CITY": 51,
            "MOSSDEEP_CITY": 52,
            "SOOTOPOLIS_CITY": 53,
            "EVER_GRANDE_CITY": 54,
            "PACIFIDLOG_TOWN": 55,

            "MT_CHIMNEY": 60,
            "SAFARI_ZONE": 61,
            "BATTLE_FRONTIER": 62,
            "SOUTHERN_ISLAND": 63,
        }


        self.layout = [
            [self.R["NONE"], self.R["ROUTE_114"], self.R["ROUTE_114"], self.R["FALLARBOR_TOWN"], self.R["ROUTE_113"], self.R["ROUTE_113"], self.R["ROUTE_113"], self.R["ROUTE_113"], self.R["ROUTE_111"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_119"], self.R["FORTREE_CITY"], self.R["ROUTE_120"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
            [self.R["NONE"], self.R["ROUTE_114"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["MT_CHIMNEY"], self.R["MT_CHIMNEY"], self.R["ROUTE_111"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_119"], self.R["NONE"], self.R["ROUTE_120"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_115"], self.R["ROUTE_114"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["MT_CHIMNEY"], self.R["MT_CHIMNEY"], self.R["ROUTE_111"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_119"], self.R["NONE"], self.R["ROUTE_120"], self.R["NONE"], self.R["NONE"], self.R["SAFARI_ZONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_115"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["LAVARIDGE_TOWN"], self.R["ROUTE_112"], self.R["ROUTE_112"], self.R["ROUTE_111"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_119"], self.R["NONE"], self.R["ROUTE_120"], self.R["ROUTE_121"], self.R["ROUTE_121"], self.R["ROUTE_121"], self.R["ROUTE_121"], self.R["LILYCOVE_CITY"], self.R["LILYCOVE_CITY"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_125"], self.R["ROUTE_125"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_115"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_111"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_119"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_122"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_125"], self.R["ROUTE_125"], self.R["NONE"], self.R["NONE"]],
            [self.R["RUSTBORO_CITY"], self.R["ROUTE_116"], self.R["ROUTE_116"], self.R["ROUTE_116"], self.R["ROUTE_116"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_111"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_119"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_122"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["ROUTE_124"], self.R["MOSSDEEP_CITY"], self.R["MOSSDEEP_CITY"], self.R["NONE"], self.R["NONE"]],
            [self.R["RUSTBORO_CITY"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["VERDANTURF_TOWN"], self.R["ROUTE_117"], self.R["ROUTE_117"], self.R["ROUTE_117"], self.R["MAUVILLE_CITY"], self.R["MAUVILLE_CITY"], self.R["ROUTE_118"], self.R["ROUTE_118"], self.R["ROUTE_123"], self.R["ROUTE_123"], self.R["ROUTE_123"], self.R["ROUTE_123"], self.R["ROUTE_123"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_126"], self.R["ROUTE_126"], self.R["ROUTE_126"], self.R["ROUTE_127"], self.R["ROUTE_127"], self.R["ROUTE_127"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_104"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_110"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_126"], self.R["SOOTOPOLIS_CITY"], self.R["ROUTE_126"], self.R["ROUTE_127"], self.R["ROUTE_127"], self.R["ROUTE_127"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_104"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_103"], self.R["ROUTE_103"], self.R["ROUTE_103"], self.R["ROUTE_103"], self.R["ROUTE_110"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_126"], self.R["ROUTE_126"], self.R["ROUTE_126"], self.R["ROUTE_127"], self.R["ROUTE_127"], self.R["ROUTE_127"], self.R["NONE"], self.R["EVER_GRANDE_CITY"]],
            [self.R["ROUTE_104"], self.R["PETALBURG_CITY"], self.R["ROUTE_102"], self.R["ROUTE_102"], self.R["OLDALE_TOWN"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_110"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_128"], self.R["ROUTE_128"], self.R["ROUTE_128"], self.R["ROUTE_128"], self.R["EVER_GRANDE_CITY"]],
            [self.R["ROUTE_105"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_101"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["SLATEPORT_CITY"], self.R["ROUTE_134"], self.R["ROUTE_134"], self.R["ROUTE_134"], self.R["ROUTE_133"], self.R["ROUTE_133"], self.R["ROUTE_133"], self.R["ROUTE_132"], self.R["ROUTE_132"], self.R["PACIFIDLOG_TOWN"], self.R["ROUTE_131"], self.R["ROUTE_131"], self.R["ROUTE_131"], self.R["ROUTE_130"], self.R["ROUTE_130"], self.R["ROUTE_130"], self.R["ROUTE_129"], self.R["ROUTE_129"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_105"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["LITTLEROOT_TOWN"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["SLATEPORT_CITY"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_105"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_109"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["BATTLE_FRONTIER"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
            [self.R["ROUTE_106"], self.R["ROUTE_106"], self.R["ROUTE_106"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["ROUTE_109"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
            [self.R["NONE"], self.R["NONE"], self.R["DEWFORD_TOWN"], self.R["ROUTE_107"], self.R["ROUTE_107"], self.R["ROUTE_107"], self.R["ROUTE_108"], self.R["ROUTE_108"], self.R["ROUTE_109"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["SOUTHERN_ISLAND"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"], self.R["NONE"]],
        ]



    def build_graph(self):
        h, w = len(self.layout), len(self.layout[0])
        graph = defaultdict(set)

        for y in range(h):
            for x in range(w):
                a = self.layout[y][x]
                if a == 0:
                    continue

                for dx, dy in [(1,0), (-1,0), (0,1), (0,-1)]:
                    nx, ny = x+dx, y+dy
                    if 0 <= nx < w and 0 <= ny < h:
                        b = self.layout[ny][nx]
                        if b != 0 and b != a:
                            graph[a].add(b)

        return graph
    

class MapNode :

    def __init__(self,id):

        self.mapId = id
        self.visited = {}

    def update_visit(self, x, y):
        if y in self.visited.get(x, []):
            return 0

        self.visited.setdefault(x, []).append(y)
        return 1
         
