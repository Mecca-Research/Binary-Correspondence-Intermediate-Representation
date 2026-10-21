from experiment import *
from dataclasses import replace
ARCH={
 'global':(Recipe(kind='global',ff=257),'fast'),
 'lowrank32':(Recipe(kind='global',ff=257,bottleneck=32),'lowrank32'),
 'lowrank32_budget':(Recipe(kind='global',ff=304,bottleneck=32),'lowrank32'),
 'multi12':(Recipe(kind='global',ff=257,rank=24,subspaces=2),'multi12'),
 'multi12_budget':(Recipe(kind='global',ff=289,rank=24,subspaces=2),'multi12'),
 'attention':(Recipe(kind='transformer',ff=256,rope=True),'attention'),
}
EXACT=('baseline','reuse','fast','compact','frame_remat','ff_remat','frame_ff_remat','frame_ff_compact','layer_remat','layer_compact')
SEEDS=(20261017,20261018,20261019)
