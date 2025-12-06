import os
from pathlib import Path
from typing import Union
import logging
import tempfile

logger = logging.getLogger(__name__)

def validate_path(path: Union[str, Path], base_dir: Union[str, Path] = None) -> Path:
    """
    Validate that a path is safe and within the expected base directory.
    Prevents path traversal attacks.

    Args:
        path (Union[str, Path]): The path to validate.
        base_dir (Union[str, Path], optional): Restricted base directory. 
                                               Defaults to project root.

    Returns:
        Path: The resolved absolute path.

    Raises:
        ValueError: If path is unsafe or outside base directory (and not in temp).
    """
    target = Path(path).resolve()
    
    # Default allowed bases: Project Root and System Temp
    allowed_bases = []
    
    if base_dir:
        allowed_bases.append(Path(base_dir).resolve())
    else:
        # Assuming src/utils.py -> project_root/src/utils.py -> project_root
        allowed_bases.append(Path(__file__).resolve().parent.parent)
        
    # Also allow system temp dir (for tests and temporary processing)
    allowed_bases.append(Path(tempfile.gettempdir()).resolve())

    is_safe = False
    for base in allowed_bases:
        try:
            # is_relative_to (Python 3.9+)
            if target.is_relative_to(base):
                is_safe = True
                break
        except AttributeError:
             # Fallback
             try:
                 target.relative_to(base)
                 is_safe = True
                 break
             except ValueError:
                 continue
    
    if not is_safe:
         msg = f"Security Error: Path {target} is outside allowed directories: {allowed_bases}"
         logger.error(msg)
         raise ValueError(msg)

    return target

def validate_array_input(data, name="Input", min_len=1, allow_empty=False):
    """
    Validate standard numpy array inputs for models/metrics.
    """
    import numpy as np
    
    # Check emptiness
    if data is None:
        raise ValueError(f"{name} cannot be None")
    
    data = np.asarray(data)
    
    if not allow_empty and data.size == 0:
        raise ValueError(f"{name} cannot be empty")
        
    if len(data) < min_len:
        raise ValueError(f"{name} length ({len(data)}) is less than minimum ({min_len})")
        
    return data
