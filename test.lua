threads = require 'threads'
dofile('dataloader.lua')

threads.serialization('threads.sharedserialize')

dataPath = './image/train/'
listPath = './image/train.txt'
local dt = DataLoader(dataPath, listPath, 32)

horses = threads.Threads(2,
    function()
        require 'torch'
    end,
    function(idx)
        print('init thread '..idx)
        dofile('dataloader.lua')
    end
)

timer = torch.Timer()
t1 = timer:time()
for i = 1,100 do
    horses:addjob(
        function()
            local x,y = dt:sample(128)
        end
    )
end

horses:synchronize()

t2 = timer:time()
print('time: '..(t2.real - t1.real))


dofile('horse.lua')
