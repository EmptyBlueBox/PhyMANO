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

#### 2. Visualization

MacOS:

```bash
mjpython mjcf_viz.py
```

### Output Files

Generated files will be saved in the following locations:
- **MJCF File**: `Models/mjcf/hand.xml`
- **Mesh Files**: `Models/mjcf/mesh/submesh_*.obj`

## MANO-URDF Integration

### File Structure
1. **`urdf_generate.py`** - URDF File Generation
   - Generate MANO hand meshes and submeshes
   - Calculate physical properties (mass, inertia tensors, etc.)
   - Create URDF XML configuration files
   - Save mesh resource files (OBJ format)

### Usage

#### 1. Generate URDF Model Files

```bash
python urdf_generate.py
```

#### 2. Visualization

Install [Rerun urdf plugin](https://github.com/rerun-io/rerun-loader-python-example-urdf):

```bash
pipx install git+https://github.com/rerun-io/rerun-loader-python-example-urdf.git
pipx ensurepath
```

Open Rerun session and `OPEN` the URDF file:

```bash
rerun
```

> [!NOTE]
> This version of Rerun is compatible with the rerun urdf plugin.
> ```bash
> rerun --version
> rerun-cli 0.19.0 [rustc 1.79.0 (129f3b996 2024-06-10), LLVM 18.1.7] aarch64-apple-darwin release-0.19.0 5efb166, built 2024-10-17T14:03:21Z
> Video features: av1 nasm
> ```

### Output Files

Generated files will be saved in the following locations:
- **URDF File**: `Models/urdf/hand.urdf`
- **Mesh Files**: `Models/urdf/mesh/submesh_*.obj`
