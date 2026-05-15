--Load functions
package.cpath = package.cpath .. ";/home/pmv/Desktop/RL/CustomBiz/BizHawk/output/Lua/RL/?.so"
mmap = require('myshm')
shm = mmap.create_shm()
input_map = {[0] = 'Up',[1] = 'Down',[2] = 'Left',[3] = 'Right',[4] = 'A',[5] = 'B',[6] = 'Start',[7] = 'load'}
frames = 0
inputs = {
    Up=false, Down=false, Left=false, Right=false,
    A=false, B=false, Start=false, Select=false
}
input = nil
while true do
    inputs.Up = false
    inputs.Down = false
    inputs.Left = false
    inputs.Right = false
    inputs.A = false
    inputs.B = false
    inputs.Start = false
    inputs.Select = false
    
    if frames == 0 then 
        memory.usememorydomain("EWRAM")
        local ewram_map = memory.readbyterange_raw(0x0,256 * 1024)
        memory.usememorydomain("IWRAM")
        local iwram_map = memory.readbyterange_raw(0x0, 32 * 1024)
        shm:write(ewram_map,iwram_map)
        input = shm:read()
    end
    if input ~= nil then
        inputs[input_map[input]] = true
    end
    if input == 7 then 
            savestate.loadslot(10)
    else 
        joypad.set(inputs)
        frames = (frames + 1) % 30
    end
    emu.frameadvance()
end
