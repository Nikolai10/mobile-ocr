# global configs

class ArchitectureDefaultConfig:
    input_shape_rgb=(384, 256, 3)
    input_shape_gray=(384, 256, 1)
    epochs=150
    batch_size=16
    optimizer='adam'
    loss='mae'