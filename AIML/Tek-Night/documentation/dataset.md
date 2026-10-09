# Dataset source
```
DIV2K Dataset
```

- 800 high resolution images were used from the DIV2K datset.

The images were split into train, validation and test accordingly:

Table:

| Split | Images |
|-------|-------:|
| Train | 700 |
| Validation | 50 |
| Test | 50 |

### Degradation pipeline:

HR
 |
Gaussian Blur
 |
Downsample ×4
 |
JPEG Compression
 |
Gaussian Noise
 |
LR