"""GROUP-ROWS-0: a copy of the tensor unit whose GEMM group jobs without a chain (the forward's q/k/v
and gate/up) deal their calls' tasks a row range at a time -- every call's column ranges of rows
[128 rho, 128 rho + 128) before the next range's -- so a worker's block of the job covers the rows
its blocks of the row-partitioned jobs around it cover. Each task is the same tile of the same call;
only which thread runs it changes. usage: proto_group_rows.py SRC/bcir_tensor.c DST/bcir_tensor.c"""

import sys
from pathlib import Path

src = Path(sys.argv[1]).read_text(encoding="utf-8")


def once(text, old, new):
    if text.count(old) != 1:
        raise SystemExit(f"anchor found {text.count(old)} times: {old[:70]!r}")
    return text.replace(old, new)


src = once(
    src,
    """  size_t nx,split,lead,total,start[BCIR_TENSOR_MM_GROUP+1];
} mm_group_job;""",
    """  size_t nx,split,lead,total,start[BCIR_TENSOR_MM_GROUP+1];
  size_t ny,rows,cum[BCIR_TENSOR_MM_GROUP+1];  /* GROUP-ROWS-0: rows nonzero, a row range at a time */
} mm_group_job;""",
)
src = once(
    src,
    """  task-=j->split;
  for (q=0;task>=j->start[q+1];q++) {}
  mm_task(&j->y[q],task-j->start[q]);
}""",
    """  task-=j->split;
  if (j->rows) {
    size_t c=task%j->cum[j->ny],rho=task/j->cum[j->ny];
    for (q=0;c>=j->cum[q+1];q++) {}
    mm_task(&j->y[q],rho*j->y[q].cols+(c-j->cum[q]));
    return;
  }
  for (q=0;task>=j->start[q+1];q++) {}
  mm_task(&j->y[q],task-j->start[q]);
}""",
)
src = once(
    src,
    """  j.total=j.split+j.start[ny]; j.lead=0;""",
    """  j.total=j.split+j.start[ny]; j.lead=0; j.ny=ny; j.rows=0; j.cum[0]=0;
  if (!nx && ny>1) {  /* GROUP-ROWS-0: every call the same row ranges */
    int same=1;
    for (q=0;q<ny;q++) { same&=j.y[q].rowsets==j.y[0].rowsets && j.start[q+1]-j.start[q]==j.y[q].cols*j.y[q].rowsets; j.cum[q+1]=j.cum[q]+j.y[q].cols; }
    if (same && j.y[0].rowsets>1) j.rows=j.y[0].rowsets;
  }""",
)
Path(sys.argv[2]).write_text(src, encoding="utf-8")
print("GROUP-ROWS-0 copy written")
