
class BattlePokemon :

    def __init__(self):
        self.hp = None
        self.status = None
        self.exp = None
    
    def update(self,hp = None,status = None,exp=None) :

        if hp is not None :
            self.hp = hp
        if status is not None:
            self.status = status
        if exp is not None :
            self.exp = exp

    def reset(self) :
        self.hp = None
        self.status = None
        self.exp = None