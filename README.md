```python
from pathlib import Path

import numpy as np
import torch
from PIL import Image

import wallsegmenter as wallseg
```


```python
device = "cuda" if torch.cuda.is_available() else "cpu"
device
```




    'cpu'




```python
weights_dir = Path("weights")
```


```python
segmenter = wallseg.build_segmenter(
    weights_dir / "resnet50_imagenet.pth",
    weights_dir / "wall_encoder_epoch_20.pth",
    weights_dir / "wall_decoder_epoch_20.pth",
    device,
    train_only_wall=True,
)
```


```python
img_path = Path("room.jpeg")
img = Image.open(img_path).convert("RGB")
pred = wallseg.segment_image(segmenter, img, device)
```

    [W NNPACK.cpp:79] Could not initialize NNPACK! Reason: Unsupported hardware.



```python
wallseg.visualize_wall(np.asarray(img), pred)
```




    
![png](wallseg_result.png)
