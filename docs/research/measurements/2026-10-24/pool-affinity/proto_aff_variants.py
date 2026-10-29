"""AFF-DESIGN: two copies of the committed unit, each without one part of POOL-AFFINITY's design, to time what
that part buys -- `nohead`: the pair and group jobs name no head, so their large weight adjoints are dealt in the
blocks with the rest; `nosteal`: no worker takes another block's leftovers, each runs its own block only. Every
value is the committed unit's (which thread runs a task never reaches the arithmetic).
usage: proto_aff_variants.py SRC NOHEAD_OUT NOSTEAL_OUT"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text()


def once(text, old, new):
    assert text.count(old) == 1, (old, text.count(old))
    return text.replace(old, new)


nohead = once(
    src,
    "pool_run_led(pool,mm_pair_task,&j,j.total,j.lead ? j.total-j.split : 0);",
    "pool_run_led(pool,mm_pair_task,&j,j.total,0);",
)
nohead = once(
    nohead,
    "pool_run_led(pool,mm_group_task,&j,j.total,j.lead ? j.total-j.split : 0);",
    "pool_run_led(pool,mm_group_task,&j,j.total,0);",
)
nosteal = once(
    src,
    "if (k && __atomic_load_n(&b->next,__ATOMIC_RELAXED)>=b->end) continue;  /* nothing left in it */",
    "if (k) break;  /* AFF-DESIGN nosteal: its own block only */",
)
Path(sys.argv[2]).write_text(nohead)
Path(sys.argv[3]).write_text(nosteal)
print("wrote", sys.argv[2], sys.argv[3])
