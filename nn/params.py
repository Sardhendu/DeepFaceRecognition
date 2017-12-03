from collections import defaultdict
moduleWeightDict = defaultdict(lambda: defaultdict())

parentPath = "/Users/sam/All-Program/App-DataSet/Deep-Neural-Nets/Models/FaceNet-Inception"

layer_name = [
    'conv1', 'bn1',
    'conv2', 'bn2',
    'conv3', 'bn3',
    'inception_3a_1x1_conv', 'inception_3a_1x1_bn',
    'inception_3a_5x5_conv1', 'inception_3a_5x5_conv2', 'inception_3a_5x5_bn1', 'inception_3a_5x5_bn2',
    'inception_3a_3x3_conv1', 'inception_3a_3x3_conv2', 'inception_3a_3x3_bn1', 'inception_3a_3x3_bn2',
    'inception_3a_pool_conv', 'inception_3a_pool_bn',
    
    'inception_3b_1x1_conv', 'inception_3b_1x1_bn',
    'inception_3b_3x3_conv1', 'inception_3b_3x3_conv2', 'inception_3b_3x3_bn1', 'inception_3b_3x3_bn2',
    'inception_3b_5x5_conv1', 'inception_3b_5x5_conv2', 'inception_3b_5x5_bn1', 'inception_3b_5x5_bn2',
    'inception_3b_pool_conv', 'inception_3b_pool_bn',
    
    'inception_3c_3x3_conv1', 'inception_3c_3x3_conv2', 'inception_3c_3x3_bn1', 'inception_3c_3x3_bn2',
    'inception_3c_5x5_conv1', 'inception_3c_5x5_conv2', 'inception_3c_5x5_bn1', 'inception_3c_5x5_bn2',
    
    'inception_4a_1x1_conv', 'inception_4a_1x1_bn',
    'inception_4a_3x3_conv1', 'inception_4a_3x3_conv2', 'inception_4a_3x3_bn1', 'inception_4a_3x3_bn2',
    'inception_4a_5x5_conv1', 'inception_4a_5x5_conv2', 'inception_4a_5x5_bn1', 'inception_4a_5x5_bn2',
    'inception_4a_pool_conv', 'inception_4a_pool_bn',

    'inception_4e_3x3_conv1', 'inception_4e_3x3_conv2', 'inception_4e_3x3_bn1', 'inception_4e_3x3_bn2',
    'inception_4e_5x5_conv1', 'inception_4e_5x5_conv2', 'inception_4e_5x5_bn1', 'inception_4e_5x5_bn2',
    
    'inception_5a_1x1_conv', 'inception_5a_1x1_bn',
    'inception_5a_3x3_conv1', 'inception_5a_3x3_conv2', 'inception_5a_3x3_bn1', 'inception_5a_3x3_bn2',
    'inception_5a_pool_conv', 'inception_5a_pool_bn',
    
    'inception_5b_1x1_conv', 'inception_5b_1x1_bn',
    'inception_5b_3x3_conv1', 'inception_5b_3x3_conv2', 'inception_5b_3x3_bn1', 'inception_5b_3x3_bn2',
    'inception_5b_pool_conv', 'inception_5b_pool_bn',
    
    'dense_layer'
]


# Following the Paper from inception module

# The first three layers are the simple conv layer followed
# conv -> Maxpool -> BN -> conv -> conv -> BN -> Maxpool
# Now the inception bloacks starts
# Block = Layer1 [1x1, 1x1, Maxpool], Layer2 [1x1, 3x3, 5x5, 1x1]
# 2* [Block -> concat] -> Maxpool
# 5 * [Block -> concat] -> Maxpool
# 2* [Block -> concat] -> Maxpool


# The Facenet model have a slightly different architecture when compared to inception modules
convShape = {
    'conv1': [64, 3, 7, 7],
    'conv2': [64, 64, 1, 1],
    'conv3': [192, 64, 3, 3],
    # Block 1
    # Chain 1
    'inception_3a_1x1_conv': [64, 192, 1, 1],
    # Chain 2
    'inception_3a_3x3_conv1': [96, 192, 1, 1],
    'inception_3a_3x3_conv2': [128, 96, 3, 3],
    # Chain 3
    'inception_3a_5x5_conv1': [16, 192, 1, 1],
    'inception_3a_5x5_conv2': [32, 16, 5, 5],
    # Chain 4
    'inception_3a_pool_conv': [32, 192, 1, 1],
    
    # Block 2
    # Chain 1
    'inception_3b_1x1_conv': [64, 256, 1, 1],
    # Chain 2
    'inception_3b_3x3_conv1': [96, 256, 1, 1],
    'inception_3b_3x3_conv2': [128, 96, 3, 3],
    # Chain 3
    'inception_3b_5x5_conv1': [32, 256, 1, 1],
    'inception_3b_5x5_conv2': [64, 32, 5, 5],
    # Chain 4
    'inception_3b_pool_conv': [64, 256, 1, 1],
    
    # Block 3  [Note: No chain 1 and 4]
    # Chain 2
    'inception_3c_3x3_conv1': [128, 320, 1, 1],
    'inception_3c_3x3_conv2': [256, 128, 3, 3],
    # Chain 3 :
    'inception_3c_5x5_conv1': [32, 320, 1, 1],
    'inception_3c_5x5_conv2': [64, 32, 5, 5],
    
    # Block 4
    # Chain 1
    'inception_4a_1x1_conv': [256, 640, 1, 1],
    # Chain 2
    'inception_4a_3x3_conv1': [96, 640, 1, 1],
    'inception_4a_3x3_conv2': [192, 96, 3, 3],
    # Chain 3
    'inception_4a_5x5_conv1': [32, 640, 1, 1 ,],
    'inception_4a_5x5_conv2': [64, 32, 5, 5],
    # Chain 4
    'inception_4a_pool_conv': [128, 640, 1, 1],
    
    # Block 5
    # Chain 2
    'inception_4e_3x3_conv1': [160, 640, 1, 1],
    'inception_4e_3x3_conv2': [256, 160, 3, 3],
    # Chain 3
    'inception_4e_5x5_conv1': [64, 640, 1, 1],
    'inception_4e_5x5_conv2': [128, 64, 5, 5],
    
    # Block 6
    # Chain 1
    'inception_5a_1x1_conv': [256, 1024, 1, 1],
    # Chain 2
    'inception_5a_3x3_conv1': [96, 1024, 1, 1],
    'inception_5a_3x3_conv2': [384, 96, 3, 3],
    # Chain 3
    'inception_5a_pool_conv': [96, 1024, 1, 1],
    
    # Bloack 7
    # Chain 1
    'inception_5b_1x1_conv': [256, 736, 1, 1],
    # Chain 2
    'inception_5b_3x3_conv1': [96, 736, 1, 1],
    'inception_5b_3x3_conv2': [384, 96, 3, 3],
    # Chain 3
    'inception_5b_pool_conv': [96, 736, 1, 1],
    
}