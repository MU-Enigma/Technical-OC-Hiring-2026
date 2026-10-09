## Quantitative results


| Model | Parameters | PSNR | SSIM | Inference Time |
|---|---:|---:|---:|---:|
| Bicubic | 0 | 23.9695 | 0.6266 | N/A |
| SRCNN | 138,947 | 24.8741 | 0.6688 | 0.8351 ms |
| ESPCN | 180,208 | 24.3840 | 0.6483 | 0.6551 ms |

## Observations

- Bicubic got a higher PSNR/SSIM 

- SRCNN achieved better reconstruction quality than ESPCN due to refining images in high-res space, but requires more computation

- ESPCN provides faster inference by performing feature extraction in low-res space and using pixelshuffle for upscaling.

- Neural networks could achieve higher scores with larger datasets, deeper architectures, longer training and more advanced loss functions

## Sources

- https://arxiv.org/ for research papers 
- AI for helping with benchmark, suggesting improvements and simple code debugging.
