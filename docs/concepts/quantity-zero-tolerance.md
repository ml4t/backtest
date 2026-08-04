# Quantity-zero tolerance

When a fill closes a position the engine must decide whether the resulting
quantity is *zero*. Floating-point arithmetic rarely lands on exactly `0.0`, so
that decision needs a tolerance. This page states the contract that tolerance
obeys.

The implementation is `quantity_zero_tolerance()` in
`ml4t/backtest/core/shared.py`.

## The arithmetic that creates a residual

A fill updates an existing position with one addition:

```python
old_qty = pos.quantity
new_qty = old_qty + ctx.signed_qty        # execution/fill_executor.py
```

When the close is economically exact, `old_qty` and `ctx.signed_qty` are equal
and opposite. Their magnitudes are then within a factor of two of each other, so
by Sterbenz's lemma **the addition itself is exact**. It contributes no error.

A non-zero `new_qty` therefore means the two operands were not exact negatives as
float64 values. They were produced by different routes — the book quantity by
accumulating prior fills, the closing quantity by sizing an order — and those
routes round differently in the last bits. The residual is the gap between them.

That gap is bounded by the spacing of float64 **at the magnitude of the
quantities being cancelled**. It is not bounded by any fixed absolute quantity.

## Controlling scale

The scale is `max(|old_qty|, |signed_qty|)`: the magnitudes actually
participating in the cancellation.

It is emphatically **not** `|new_qty|`. `new_qty` is approximately zero by
construction, so a tolerance derived from it would be approximately zero as well:
`abs(new_qty) < k * ulp(abs(new_qty))` is false for every `k < 1` and every
normal float. A relative tolerance taken against zero is mathematically
ineffective, which is why the tolerance takes the *operands* rather than the
result.

`max()` rather than either operand alone, because the two are interchangeable in
the closure case and `max()` is the only choice that is symmetric in them. It is
also the correct upper bound on the last-place spacing of both.

## Why the previous absolute epsilon failed

The engine previously compared against an absolute `1e-12`. Float64 spacing grows
with magnitude, so a single absolute threshold can only be correct over one
binade:

| position size | one ULP | caught by `1e-12`? |
|---|---|---|
| 4096 | 9.094947e-13 | yes |
| 8191 | 9.094947e-13 | yes |
| **8192** | **1.818989e-12** | **no** |
| 10927 | 1.818989e-12 | no |
| 874442 | 1.164153e-10 | no |

At exactly 2<sup>13</sup> = 8192 units the spacing crosses `1e-12` and the rule
stops working. From there upward a one-ULP residual survives, the position key is
retained, and the engine reports a position that holds a quantity of the order of
1e-12 shares.

A **larger** fixed epsilon is not the repair. It moves the failure point to a
larger magnitude without removing it — every absolute threshold is scale-blind
somewhere — and it erases genuine small positions below the new threshold. The
defect is the absoluteness, not the value.

For the same reason the tolerance carries **no upper cap**. A ceiling is itself a
fixed absolute value, so capping would reintroduce exactly the scale-blindness
being removed.

## The rule

```
scale = max(|operand|) over the finite operands
tol   = QTY_ZERO_FLOOR                       if scale == 0
        max(QTY_ZERO_FLOOR, QTY_ZERO_ULPS * ulp(scale))   otherwise

QTY_ZERO_FLOOR = 1e-12
QTY_ZERO_ULPS  = 16
```

A quantity counts as zero when `|q| <= tol`, and is open when `|q| > tol`. That
single predicate holds at every site. The engine previously mixed `<` and `<=`
against its absolute epsilon, so its sites disagreed with each other exactly at
the threshold; they are unified here on the `<=` form the majority already used.

### Why 16 ULP

* **Mechanism.** The residual is the last-bits gap between the book quantity and
  the order quantity. Order sizing (a target weight through equity and price, or
  a re-read of the book) costs a small number of ULP; accumulating a position
  over *n* fills adds at most `n/2` ULP of drift between the book value and the
  exact sum.
* **Margin over observation.** Across every retained run available for this
  repair, every unsnapped residual is **exactly one ULP** of the operation scale.
  16 ULP is a sixteen-fold margin over the worst case actually seen, and covers
  roughly 32 accumulation steps at the worst-case half-ULP each.
* **Relative size.** 16 ULP is `16 * 2**-52 ≈ 3.6e-15` of the operation scale —
  the fifteenth significant digit. float64 carries about 15.95 significant
  decimal digits, so a distinction that small cannot be *reliably* produced by
  any float64 computation working at that scale.
* **Headroom below a submittable order.** `OrderBook._MIN_ORDER_SIZE` is `1e-8`:
  the engine refuses to submit anything smaller. At the largest operation scale
  observed in practice (874,443 units) the tolerance is 1.86e-9 — still well
  under that floor.

### Why the absolute floor

`ulp(scale)` collapses toward zero as the scale does, so a pure ULP rule would
have no useful width for small positions and none at all at `scale == 0`. The
floor keeps the historical absolute behaviour everywhere float64 spacing is finer
than `1e-12`, which is every magnitude below 8192.

This gives the migration a provable property: **a call with a single operand is
exactly equivalent to the old `abs(q) < 1e-12` rule**. For any normal `x`,
`16 * ulp(x) < x`, so the ULP term can never make such a predicate true on its
own; only the floor can. Call sites that merely ask "is this book quantity zero?"
therefore keep their previous behaviour bit for bit, while call sites that pass
both closure operands gain the scale-aware rule.

## Residue versus a genuine small position

A **residue** is what remains after an economically exact close: its magnitude is
a handful of ULP of the quantities that cancelled.

A **genuine** small position is one a strategy meant to hold. It is preserved
because the tolerance is relative to *the operation that produced it*, not to a
global constant:

* Opening 1e-9 units on an empty book is an operation at scale 1e-9, where the
  tolerance is the floor, 1e-12. The position survives.
* Holding 10,000 units and selling 9,999.999999999 leaves a genuine 1e-9. The
  operation scale is 10,000, where the tolerance is 1.46e-11. 1e-9 is seventy
  times larger, so the position survives.

The only quantity the rule erases is one arrived at by cancelling two much larger
quantities — and at that point the result is below what float64 arithmetic at
that scale can warrant.

## Edge cases

| input | behaviour |
|---|---|
| `scale == 0` (no position, zero fill) | tolerance is the floor, `1e-12` |
| subnormal operands | `ulp` is the minimum subnormal; the floor dominates |
| `inf` / `NaN` operand | ignored when computing the scale; the tolerance stays finite and positive, and falls back to the floor if no operand is finite |
| `NaN` quantity under test | `abs(nan) < tol` is false, so it is not treated as zero — unchanged from the previous rule |
| long versus short | the tolerance is built from magnitudes, so it is identical for `+q` and `-q` |

## Sites

One shared primitive serves every site that asks whether a *quantity* is zero:
position closure in the fill executor, the submission-precheck shadow book, the
shadow-queue validation and commit paths, and the gatekeeper's position
normalisation.

`OrderBook._MIN_ORDER_SIZE` is deliberately **not** migrated. It is an economic
policy — the smallest order the engine will accept — not a statement about
floating-point residue.
