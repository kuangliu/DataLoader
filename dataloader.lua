require 'xlua';
require 'image';
require 'torch';
require 'paths';
require 'os';
require 'sys';
ffi = require 'ffi';

torch.setdefaulttensortype('torch.FloatTensor')

local DataLoader = torch.class 'DataLoader'
local pathcat = paths.concat

function DataLoader:__init(dataPath, listPath, imageProcess)
    ---------------------------------------------------------------------------
    -- DataLoader takes:
    --  -  dataPath: a folder containing all the images.
    --  -  listPath: a text file containing all sample names & targets.
    --  -  imageProcess: the image process function while sampling, performing
    --                   resizing, zero-mean, normalization, auumentation.
    ---------------------------------------------------------------------------
    assert(paths.dirp(dataPath), dataPath..' not exist!')
    assert(paths.filep(listPath), listPath..' not exist!')
    self.dataPath = dataPath
    self.listPath = listPath
    self.imageProcess = imageProcess or 'defaultProcess'

    self:parseList()
end

function DataLoader:parseList()
    ---------------------------------------------------------------------------
    -- parse list files, split file names and targets.
    -- returns:
    --    - names: 2D tensor containing the names sized [N, maxNameLength]
    --    - targets: 2D tensor containing the targets sized [N, D]
    -- where
    --   - N: # of samples
    --   - maxNameLength: max length of all names
    --   - D: dims of the target
    --
    -- We first pre-allocate names sized [N, constLength],
    -- and trim to [N, maxNameLength] later.
    ---------------------------------------------------------------------------
    print('parsing list...')

    local constLength = 50           -- assume the length of all file names < constLength
    local maxNameLength = -1         -- max file name length

    -- Get the number of files
    local N = tonumber(sys.fexecute('ls '..self.dataPath..' | wc -l'))
    self.nSamples = N

    local names = torch.CharTensor(N,constLength):fill(0)
    local targets = nil

    -- Parse names and targets line by line
    local name_data = names:data()
    local f = assert(io.open(self.listPath, 'r'))
    for i = 1,N do
        xlua.progress(i,N)
        local line = f:read('*l')

        local isfirst = true
        local target = {}
        for s in string.gmatch(line, '%S+') do
            if isfirst then -- image name
                ffi.copy(name_data, s)
                name_data = name_data + constLength
                maxNameLength = math.max(maxNameLength, #s)
                isfirst = false
            else            -- targets
                target[#target+1] = tonumber(s)
            end
        end
        targets = targets or torch.Tensor(N, #target)
        targets[i] = torch.Tensor(target)
    end
    f:close()

    -- Trim names from [N,constLength] -> [N,maxNameLength]
    names = names[{ {},{1,maxNameLength} }]
    self.names = names
    self.targets = targets
end

function DataLoader:__loadImages(indices)
    --------------------------------------
    -- Load images from the given indices.
    --------------------------------------
    local quantity = indices:nElement()
    local images
    for i = 1,quantity do
        xlua.progress(i,quantity)
        local name = ffi.string(self.names[indices[i]]:data())
        local im = image.load(pathcat(self.dataPath, name))
        im = self.imageProcess(im)   -- recale, zero-mean & normalization

        images = images or torch.Tensor(quantity, 3, im:size(2), im:size(2))
        images[i] = im
    end
    return images
end

function DataLoader:sample(quantity)
    ---------------------------------------------------------
    -- Randomly sample quantity images from training dataset.
    -- Load samples maybe overlapped.
    ---------------------------------------------------------
    print('loading a batch of samples...')

    assert(quantity, 'No sample quantity specified!')
    local indices = torch.LongTensor(quantity):random(self.nSamples)
    local images  = self:__loadImages(indices)
    local targets = self.targets:index(1,indices)
    return images, targets
end

function DataLoader:loadBatchByIndex(quantity, index, seed)
    ------------------------------------------------------------
    -- 1. randomly shuffle all the indices by the given seed
    -- 2. split the shuffled indices by the quantity
    -- 3. return the index batch
    --
    -- In the same epoch, the random seed won't change, so the
    -- generated indices won't change, and each sample will be
    -- loaded once, no overlapping (unlike `sample` method above)
    ------------------------------------------------------------
    -- Shuffle all the indices
    torch.manualSeed(seed)
    local indices = torch.randperm(self.nSamples):long():split(quantity)

    -- Get the index-th batch
    local inds = indices[index]
    local images = self:__loadImages(inds)
    local targets = self.targets:index(1,inds)
    return images, targets
end

function DataLoader:get(i1, i2)
    ------------------------------------------------
    -- Get images in the index range [i1, i2]
    -- Used to load test samples.
    ------------------------------------------------
    local indices = torch.range(i1, i2):long()
    local images = self:__loadImages(indices)
    local targets = self.targets:index(1,indices)
    return images, targets
end
