import os
from pathlib import Path

config = dict(
    data_dir = Path('/home/matyushinleonid/data/projects/PID_Compression/data/'),

    classification = dict(
        val_size=0.2,
        test_size=0.2,
        random_state=555,
        X_columns=['S5aux0', 'S3aux0', 'S2aux0', 'S0aux0', 'S0aux1', 'S0aux2',
       'S0aux3', 'S2aux1', 'S2aux2', 'S2aux3', 'S0aux4', 'S0aux5', 'S0aux6',
       'S0aux7', 'S0aux8', 'S0x0', 'S0x1', 'S0x2', 'S0x3', 'S0x4', 'S3x0',
       'S3x1', 'S2x0', 'S2x1', 'S2x2', 'S2x3', 'S0x5', 'S0x6', 'S0x7', 'S0x8',
       'S0x9', 'S0x10', 'S1x0', 'S1x1', 'S1x2', 'S1x3', 'S1x4', 'S1x5', 'S5x0',
       'S4x0', 'S4x1', 'S4x2', 'S3x2', 'S4x3', 'S4x4', 'S5x1', 'S5x2', 'S5x3',
       'S5x4', 'S4x5'],
        target_column='pid'
    )
)

if not os.path.isdir(config['data_dir']):
    os.mkdir(config['data_dir'])

for suf in ['cache', 'output']:
    if not os.path.isdir(config['data_dir'] / suf):
        os.mkdir(config['data_dir'] / suf)

if not os.path.isdir(config['data_dir'] / 'cache' / 'initial_data_split'):
    os.mkdir(config['data_dir'] / 'cache' / 'initial_data_split')

for suf in ['cache', 'output']:
    if not os.path.isdir(config['data_dir'] / suf / 'models'):
        os.mkdir(config['data_dir'] / suf / 'models')

