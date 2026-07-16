/*===- bcir_train.c - the D1 numeric stage kernels (see bcir_train.h) ---------------------===
 *
 * Kernel-for-kernel twin of bcir/kbcir/train_graph.py `_step_kernels`. Every sum accumulates
 * in ascending index order (Python's generator sums), the sigmoid keeps the oracle's guarded
 * two-branch form, and the BCE keeps the eps clamp -- so on a shared libm the loss curve and
 * the trained weights match the oracle to the last bit (the differential gate allows only
 * float round-off for cross-libm CI).
 *===----------------------------------------------------------------------===*/
#include "bcir_train.h"

#include <limits.h>
#include <math.h>

static int tr_state_valid(const bcir_train_state *st) {
  if (!st || st->nf <= 0 || st->nf == INT_MAX || st->b <= 0 || !isfinite(st->lr) ||
      !st->X || !st->y || !st->w || !st->z || !st->act || !st->lossv || !st->grad)
    return 0;
  return (size_t)st->nf <= SIZE_MAX / (size_t)st->b;
}

static int stream_state_valid(const bcir_stream_state *st) {
  size_t rows, weights;
  if (!st || st->nf <= 0 || st->nf == INT_MAX || st->mb <= 0 || st->streams <= 0 ||
      !isfinite(st->lr) || !st->X || !st->y || !st->w || !st->z || !st->act ||
      !st->lossv || !st->loss || !st->grad || !st->gradc)
    return 0;
  if ((size_t)st->streams > SIZE_MAX / (size_t)st->mb) return 0;
  rows = (size_t)st->streams * (size_t)st->mb;
  if ((size_t)st->nf > SIZE_MAX / rows) return 0;
  weights = (size_t)st->nf + 1u;
  return weights <= SIZE_MAX / (size_t)st->streams;
}

/* The oracle's guarded sigmoid (kbcir/recurrent.py): for x >= 0 use 1/(1+exp(-x)) (exp <= 1,
 * no overflow); else the algebraically-equal exp(x)/(1+exp(x)). */
static double tr_sigmoid(double x) {
  if (x >= 0.0) { double e = exp(-x); return 1.0 / (1.0 + e); }
  double e = exp(x);
  return e / (1.0 + e);
}

static void tr_forward(bcir_train_state *st) {            /* claim 1: z = X @ w + bias */
  for (int i = 0; i < st->b; i++) {
    double acc = 0.0;
    for (int j = 0; j < st->nf; j++) acc += st->X[i * st->nf + j] * st->w[j];
    st->z[i] = acc + st->w[st->nf];
  }
}

static void tr_activation(bcir_train_state *st) {         /* claim 2: act = sigmoid(z) */
  for (int i = 0; i < st->b; i++) st->act[i] = tr_sigmoid(st->z[i]);
}

static void tr_loss_vec(bcir_train_state *st) {           /* claim 3: per-example BCE */
  const double eps = 1e-12;
  for (int i = 0; i < st->b; i++) {
    double a = st->act[i];
    if (a < eps) a = eps;
    if (a > 1.0 - eps) a = 1.0 - eps;
    double yv = st->y[i];
    st->lossv[i] = -(yv * log(a) + (1.0 - yv) * log(1.0 - a));
  }
}

static void tr_reduce_mean(bcir_train_state *st) {        /* claim 4: loss = mean(lossv) */
  double acc = 0.0;
  for (int i = 0; i < st->b; i++) acc += st->lossv[i];
  st->loss = acc / st->b;
}

static void tr_backward(bcir_train_state *st) {           /* claim 5: exact BCE+sigmoid grad */
  for (int j = 0; j < st->nf; j++) {
    double acc = 0.0;
    for (int i = 0; i < st->b; i++) acc += (st->act[i] - st->y[i]) * st->X[i * st->nf + j];
    st->grad[j] = acc / st->b;
  }
  double acc = 0.0;
  for (int i = 0; i < st->b; i++) acc += st->act[i] - st->y[i];
  st->grad[st->nf] = acc / st->b;
}

static void tr_update(bcir_train_state *st) {             /* claim 6: sgd step */
  for (int j = 0; j <= st->nf; j++) st->w[j] -= st->lr * st->grad[j];
}

int bcir_train_kernel(const bcir_exec_item *item, void *ctx) {
  bcir_train_state *st = (bcir_train_state *)ctx;
  if (!item) return 1;
  if (item->claim_id < 1 || item->claim_id > 6) return 0;
  if (!tr_state_valid(st)) return 1;
  switch (item->claim_id) {
  case 1: tr_forward(st); break;
  case 2: tr_activation(st); break;
  case 3: tr_loss_vec(st); break;
  case 4: tr_reduce_mean(st); break;
  case 5: tr_backward(st); break;
  case 6: tr_update(st); break;
  default: break;                        /* the oracle's kernels.get(id) miss: a no-op */
  }
  return 0;
}

/* --- D1 step 7: the streamed step's kernels (train_stream_module's claim-id bands) ------- */

static void sk_forward(bcir_stream_state *st, int s) {     /* s*10+1: z_s = X_s @ w + bias */
  for (int i = 0; i < st->mb; i++) {
    double acc = 0.0;
    const double *row = st->X + ((size_t)s * st->mb + i) * st->nf;
    for (int j = 0; j < st->nf; j++) acc += row[j] * st->w[j];
    st->z[(size_t)s * st->mb + i] = acc + st->w[st->nf];
  }
}

static void sk_activation(bcir_stream_state *st, int s) {  /* s*10+2 */
  for (int i = 0; i < st->mb; i++) {
    size_t o = (size_t)s * st->mb + i;
    st->act[o] = tr_sigmoid(st->z[o]);
  }
}

static void sk_loss_vec(bcir_stream_state *st, int s) {    /* s*10+3: per-example BCE */
  const double eps = 1e-12;
  for (int i = 0; i < st->mb; i++) {
    size_t o = (size_t)s * st->mb + i;
    double a = st->act[o];
    if (a < eps) a = eps;
    if (a > 1.0 - eps) a = 1.0 - eps;
    double yv = st->y[o];
    st->lossv[o] = -(yv * log(a) + (1.0 - yv) * log(1.0 - a));
  }
}

static void sk_reduce_mean(bcir_stream_state *st, int s) { /* s*10+4: loss_s = mean */
  double acc = 0.0;
  for (int i = 0; i < st->mb; i++) acc += st->lossv[(size_t)s * st->mb + i];
  st->loss[s] = acc / st->mb;
}

static void sk_backward(bcir_stream_state *st, int s) {    /* s*10+5: the stream's mean grad */
  double *g = st->grad + (size_t)s * (st->nf + 1);
  for (int j = 0; j < st->nf; j++) {
    double acc = 0.0;
    for (int i = 0; i < st->mb; i++) {
      size_t o = (size_t)s * st->mb + i;
      acc += (st->act[o] - st->y[o]) * st->X[o * st->nf + j];
    }
    g[j] = acc / st->mb;
  }
  double acc = 0.0;
  for (int i = 0; i < st->mb; i++) {
    size_t o = (size_t)s * st->mb + i;
    acc += st->act[o] - st->y[o];
  }
  g[st->nf] = acc / st->mb;
}

static void sk_combine(bcir_stream_state *st) {            /* streams*10+1: mean of means */
  for (int j = 0; j <= st->nf; j++) {
    double acc = 0.0;
    for (int s = 0; s < st->streams; s++) acc += st->grad[(size_t)s * (st->nf + 1) + j];
    st->gradc[j] = acc / st->streams;
  }
}

static void sk_update(bcir_stream_state *st) {             /* streams*10+2: the SINGLE step */
  for (int j = 0; j <= st->nf; j++) st->w[j] -= st->lr * st->gradc[j];
}

int bcir_stream_kernel(const bcir_exec_item *item, void *ctx) {
  bcir_stream_state *st = (bcir_stream_state *)ctx;
  if (!item) return 1;
  if (!stream_state_valid(st)) return 1;
  uint64_t id = item->claim_id;
  uint64_t terminal = (uint64_t)st->streams * 10u;
  if (id == terminal + 1u) { sk_combine(st); return 0; }
  if (id == terminal + 2u) { sk_update(st); return 0; }
  uint64_t stream = id / 10u;
  int k = (int)(id % 10u);
  if (stream >= (uint64_t)st->streams || k < 1 || k > 5) return 0; /* kernels.get miss */
  int s = (int)stream;
  switch (k) {
  case 1: sk_forward(st, s); break;
  case 2: sk_activation(st, s); break;
  case 3: sk_loss_vec(st, s); break;
  case 4: sk_reduce_mean(st, s); break;
  case 5: sk_backward(st, s); break;
  }
  return 0;
}
