// Float rounding shared by the ported decoder (FFmpeg commit 2da55bf, as
// its arm64 build compiles libavcodec/opus/silk.c, celt.h and pvq.c).
// LGPL-2.1-or-later (see LICENSE-LGPL).

/// `sum + Σ a_k b_k` for `k < n`, `term(k) = (a_k, b_k)`, rounded the way
/// FFmpeg's arm64 build (clang, `-ffp-contract=on`) evaluates an in-order
/// reduction loop such as `sum += a[k] * b[k]`: clang vectorizes it in
/// blocks of four products, each rounded and then added in `k` order, and
/// leaves the last `n % 4` terms to fused multiply-adds (with `n < 4`, every
/// term fuses). Used for silk.c's LPC loops, `celt_renormalize_vector`'s sum
/// of squares and `celt_stereo_merge`'s two sums, whose objects show this
/// shape; a last-bit difference there carries into the output.
#[inline]
pub(super) fn ordered_dot(mut sum: f32, n: usize, term: impl Fn(usize) -> (f32, f32)) -> f32 {
    let blocked = n & !3;
    for k in 0..blocked {
        let (a, b) = term(k);
        sum += a * b;
    }
    for k in blocked..n {
        let (a, b) = term(k);
        sum = a.mul_add(b, sum);
    }
    sum
}
