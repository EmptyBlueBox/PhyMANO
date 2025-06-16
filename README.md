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

The project includes a specialized MuJoCo integration that has been split into two independent modules for ease of use and maintenance:

### File Structure

1. **`mjcf_generate.py`** - MJCF File Generation
   - Contains all functions related to MANO mesh generation and MJCF file creation
   - Can be run directly to generate hand model files

2. **`mjcf_viz.py`** - Visualization Testing
   - Loads and visualizes saved MJCF files
   - Provides real-time physics simulation and control testing

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

### Functionality

#### mjcf_generate.py Features:
- Generate MANO hand meshes and submeshes
- Calculate physical properties (mass, inertia tensors, etc.)
- Create MuJoCo XML configuration files
- Save mesh resource files (OBJ format)

#### mjcf_viz.py Features:
- Load saved MJCF files
- Launch MuJoCo visualizer
- Apply random control signals for physics testing
- Provide real-time visualization feedback

This design ensures that both functional modules are completely independent and can be used separately as needed.
