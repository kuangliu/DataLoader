# Torch data loading module
We use Torch tensor instead of Lua table to store file names & targets.
It's faster to access, and saves more spaces.

## Directory arrangement
-- dataPath  
|-- train  
|-- test  
|-- train.txt
|-- test.txt

- `dataPath` is root directory, and passed as argument.
- `train`&`test` are folders containing training/test images.
- `train.txt`&`test.txt` containing the file names & labels/targets separated by spaces.
