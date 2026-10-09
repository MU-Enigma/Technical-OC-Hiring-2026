# SRCNN

SRCNN (super resolution cnn) is a classic CNN baseline for image upscaling. It uses a convolutional network to learn non-linear features between low-res and high-res images.

### Architecture:

Input LR image (32x32x3)

        |
        |
Apply bicubic upscaling

        |

(128x128x3)

        |

        Conv(9x9)
        ReLU

        |
        |

        Conv(5x5)
        ReLU

        |
        |

        Conv(3x3)
        ReLU

        |
        |

        Conv(5x5)

        |
        |
Residual connection with bicubic image

        |

Output HR image
(128x128x3)


Parameters: 138,947


# ESPCN

ESPCN (efficient sub-pixel cnn) is a CNN model used to upscale low-res images to high-res images.

### Why is it better?
- It is much faster than traditional image upscaling methods as it performs all core feature extraction in the LR space, rather than on the expanded HR image.
- The upscaling step is pushed to the very last layer of the network using a sub-pixel convolutional layer. ESPCN rearranges the feature maps into a higher-res grid in one go.
- As the input size is smaller, the network requires small filter sizes to extract features; leading to much lower memory costs and computational complexity.
- Residual learning is used where the network learns high-frequency details over a bicubic baseline image.

### Architecture

Input LR image (32x32x3)

        |
        |
        Conv(5x5)
        ReLU

        |

        Conv(3x3)
        ReLU

        |

        Conv(3x3)
        ReLU

        |

        Conv(3x3)

        |

        PixelShuffle x4

        |

Residual addition with bicubic upscaled image

        |

(128x128x3)


Parameters: 180,208