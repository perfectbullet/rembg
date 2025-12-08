# Rembg AI Coding Agent Instructions

## Project Overview
Rembg is a Python tool for removing image backgrounds using ONNX-based deep learning models. It provides both a library API and CLI with multiple interfaces (file, folder, HTTP server, binary stream).

## Architecture

### Core Components
- **`rembg/bg.py`**: Main background removal logic via `remove()` function. Handles multiple input types (bytes, PIL Image, numpy array), applies models, manages alpha matting, and post-processing.
- **`rembg/session_factory.py`**: Factory pattern for model sessions. Use `new_session(model_name)` to instantiate models. Automatically configures ONNX runtime with CPU/GPU/ROCM providers.
- **`rembg/sessions/`**: Model implementations. Each session class inherits `BaseSession` and implements `predict()` and `download_models()` class methods.
- **`rembg/commands/`**: CLI commands (`i`, `p`, `s`, `b`, `d`) using Click framework.

### Model Session Pattern
All model sessions follow this structure:
- Inherit from `BaseSession` (`rembg/sessions/base.py`)
- Implement `predict(img: PILImage) -> List[PILImage]` to return mask(s)
- Implement `download_models()` classmethod using `pooch.retrieve()` for model download
- Implement `name()` classmethod returning model identifier
- Models stored in `~/.u2net/` (configurable via `U2NET_HOME`)

Example session structure (see `rembg/sessions/u2net.py`, `birefnet_general.py`):
```python
class MySession(BaseSession):
    def predict(self, img: PILImage, *args, **kwargs) -> List[PILImage]:
        ort_outs = self.inner_session.run(None, self.normalize(img, mean, std, size))
        # Process predictions and return PIL mask(s)
        return [mask]
    
    @classmethod
    def download_models(cls, *args, **kwargs):
        fname = f"{cls.name()}.onnx"
        pooch.retrieve(url, md5_checksum, fname=fname, path=cls.u2net_home())
        return os.path.join(cls.u2net_home(), fname)
```

### Session Registry
`rembg/sessions/__init__.py` maintains `sessions` dict and `sessions_names` list. Add new models here.

## Development Workflows

### Running Tests
```bash
pytest tests/test_remove.py  # Tests all models against fixtures
```
Tests use image hashing (`imagehash.phash`) to compare actual vs expected results. To update expected results, uncomment write block in `test_remove.py`.

### Installing Dev Environment
```bash
pip install ".[dev,cpu,cli]"  # or [gpu,cli] for CUDA support
```

### Building
- **PyPI package**: Uses `setup.py` with versioneer for git-based versioning
- **Docker**: `Dockerfile` for CPU, `Dockerfile_nvidia_cuda_cudnn_gpu` for GPU
- **Windows EXE**: `_build-exe.ps1` with PyInstaller (`rembg.spec`), creates Inno Setup installer (`_setup.iss`)

### Adding a New Model
1. Create `rembg/sessions/your_model.py` inheriting `BaseSession`
2. Implement `predict()`, `download_models()`, `name()` methods
3. Register in `rembg/sessions/__init__.py` sessions dict
4. Add model to CLI choices in command files
5. Add test case with fixtures/expected results

## Key Conventions

### Input/Output Handling
The `remove()` function auto-detects input type and returns matching type:
- bytes → bytes (PNG format)
- PIL Image → PIL Image  
- numpy array → numpy array
- Use `force_return_bytes=True` to always return bytes

### CLI Command Structure
Commands use Click decorators with consistent option patterns:
- `-m/--model`: Model selection (choices from `sessions_names`)
- `-a/--alpha-matting`: Enable alpha matting refinement
- `-om/--only-mask`: Return mask only
- `-x/--extras`: JSON dict for model-specific parameters (e.g., SAM prompts)

See `rembg/commands/i_command.py` for reference implementation.

### Model-Specific Parameters
Pass via `extras` in CLI (`-x '{"key": "value"}'`) or `**kwargs` in API:
```python
# SAM with point prompt
remove(img, session=new_session("sam"), 
       sam_prompt=[{"type": "point", "data": [x, y], "label": 1}])
```

### Session Reuse (Performance Critical)
Always reuse sessions when processing multiple images:
```python
session = new_session("birefnet-general")
for img in images:
    output = remove(img, session=session)  # Don't create new session per image
```

## External Dependencies

### ONNX Runtime
- Install `onnxruntime` (CPU), `onnxruntime-gpu` (CUDA), or `onnxruntime-rocm` (AMD)
- Provider selection automatic in `BaseSession.__init__()` based on `ort.get_device()`
- Override via `providers=["CUDAExecutionProvider"]` kwarg

### Model Downloads
- Uses `pooch` for cached downloads with MD5 verification
- Disable checksum via `MODEL_CHECKSUM_DISABLED` env var
- Progress bars shown during first download

### Alpha Matting
- Uses `pymatting` library for trimap-based refinement
- Computationally expensive; use sparingly
- Falls back to naive cutout on errors

## HTTP Server (`rembg s`)
FastAPI-based server on port 7000:
- `/api/remove?url=<image_url>`: Remove background from URL
- `/api/remove` (POST with file upload): Remove from uploaded image
- Gradio UI for interactive testing
- Session pooling: creates sessions on-demand, reuses across requests

## Critical Notes
- Python version support: 3.10-3.13 (limited by onnxruntime)
- Image orientation auto-fixed via EXIF (`fix_image_orientation()`)
- Post-processing: morphological operations + Gaussian blur for smooth masks
- Custom models: use `u2net_custom` or `dis_custom` sessions with `model_path` extra
