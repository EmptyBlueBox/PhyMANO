# PhyMANO

Simulate MANO in physics simulators.

## Prerequisites

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

## MANO-MuJoCo Integration

### File Structure
1. **`mjcf_generate.py`** - MJCF File Generation
   - Generate MANO hand meshes and submeshes
   - Calculate physical properties (mass, inertia tensors, etc.)
   - Create MuJoCo XML configuration files
   - Save mesh resource files (OBJ format)

2. **`mjcf_viz.py`** - Visualization Testing
   - Load saved MJCF files
   - Set joint rotations
   - Provide real-time visualization feedback

### Usage

#### 1. Generate MJCF Model Files
```bash
python mjcf_generate.py
```

#### 2. Visualization Testing

MacOS:

```bash
mjpython mjcf_viz.py
```

### Output Files

Generated files will be saved in the following locations:
- **MJCF File**: `Models/mjcf/hand.xml`
- **Mesh Files**: `Models/mjcf/mesh/submesh_*.obj`
