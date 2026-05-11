import struct

class Memory :
    '''
    Allows access of binary memory data from the ram map
    '''
    def __init__(self,shm,base):
        self.ram = shm
        self.base = base

    def read_flag(self,offset):
        return struct.unpack_from('?', self.ram, self.base + offset)[0]

    def read_u8(self,offset):
        return self.ram[self.base + offset]
    
    def read_s8(self, offset):
        return struct.unpack_from('b', self.ram, self.base + offset)[0]
    
    def read_s16_le(self,offset):
        return struct.unpack_from('<h',self.ram,self.base + offset)[0] 
    
    def read_s16_be(self,offset):
        return struct.unpack_from('>h',self.ram,self.base + offset)[0]
    
    def read_u16_le(self,offset):
        return struct.unpack_from('<H',self.ram,self.base + offset)[0]
    
    def read_u16_be(self,offset):
        return struct.unpack_from('>H',self.ram,self.base + offset)[0]

    def read_s32_le(self,offset):
        return struct.unpack_from('<i',self.ram,self.base + offset)[0]
    
    def read_s32_be(self,offset):
        return struct.unpack_from('>i',self.ram,self.base + offset)[0]
    
    def read_u32_le(self,offset):
        return struct.unpack_from('<I',self.ram,self.base + offset)[0]
    
    def read_u32_be(self,offset):
        return struct.unpack_from('>I',self.ram,self.base + offset)[0]