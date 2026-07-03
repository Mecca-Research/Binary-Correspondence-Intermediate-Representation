/*===- bcir_train.c - the D1 numeric stage kernels (see bcir_train.h) ---------------------===
 *
 * Kernel-for-kernel twin of bcir/kbcir/train_graph.py `_step_kernels`. Every sum accumulates
 * in ascending index order (Python's generator sums), the sigmoid keeps the oracle's guarded
 * two-branch form, and the BCE keeps the eps clamp -- so on a shared libm the loss curve and
 * the trained weights match the oracle to the last bit (the differential gate allows only
 * float round-off for cross-libm CI).
 *===----------------------------------------------------------------------===*/
#include "bcir_train.h"

#include <math.h>

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
