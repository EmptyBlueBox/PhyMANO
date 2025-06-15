# PhyMANO

Simulate MANO in physics simulators.

## Usage

### Downloading models

Please register on the [MANO website](https://mano.is.tue.mpg.de/) and download the models. Place the models in a folder with the following structure:

```bash
Models
|
└── mano
    ├── MANO_RIGHT.pkl
    └── MANO_LEFT.pkl
```

The MANO joint order:

```
   15 - 14 - 13 - \ Thumb
                   \
   3-- 2 -- 1 ----- 0 Index
    6 -- 5 -- 4 -- / Middle
12 -- 11 -- 10 -- / Ring
  9 -- 8 -- 7 -- / Pinky
```

### Environment setup

```bash
conda create -n phymano python=3.10
conda activate phymano
pip install -r requirements.txt
```

### Running the code

```bash
python main.py
```
