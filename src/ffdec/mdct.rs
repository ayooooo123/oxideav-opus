// The inverse MDCT FFmpeg's CELT decoder runs, as FFmpeg 2da55bf's C path
// computes it (libavutil/tx.c, tx_template.c, tx_priv.h; `-cpuflags 0`):
// `av_tx_init(AV_TX_FLOAT_MDCT, inv = 1, len, scale)` for len 120, 240, 480
// and 960 picks `mdct_pfa_15xM_inv_float_c`, a prime-factor transform of
// len/2 = 15 x M points over the split-radix `fftM_ns_float_c` (M = 4, 8,
// 16, 32): `len` coefficients in, `len` samples out (the middle half of
// the full inverse transform).
//
// Single-precision throughout, with the rounding of FFmpeg's arm64 build
// (clang, `-ffp-contract=on`): every `a*b ± c*d` in one C expression fuses
// its left product and rounds the right one; products in separate
// statements are rounded. The tables are computed in double and rounded to
// float, as FFmpeg's are.
// Copyright (c) Lynne (tx.c, tx_template.c, tx_priv.h);
// LGPL-2.1-or-later (see LICENSE-LGPL).

use std::f64::consts::{FRAC_PI_2, PI};

#[derive(Clone, Copy, Default)]
struct C32 {
    re: f32,
    im: f32,
}

/// `CMUL`: `(are*bre - aim*bim, are*bim + aim*bre)`.
#[inline]
fn cmul(are: f32, aim: f32, bre: f32, bim: f32) -> (f32, f32) {
    (are.mul_add(bre, -(aim * bim)), are.mul_add(bim, aim * bre))
}

/// `SMUL`: `(are*bre - aim*bim, are*bim - aim*bre)`.
#[inline]
fn smul(are: f32, aim: f32, bre: f32, bim: f32) -> (f32, f32) {
    (
        are.mul_add(bre, -(aim * bim)),
        are.mul_add(bim, -(aim * bre)),
    )
}

/// `ff_tx_tab_53`: the 5- and 3-point constants.
fn tab_53() -> [f32; 12] {
    let c5 = (2.0 * PI / 5.0).cos() as f32;
    let c10 = (2.0 * PI / 10.0).cos() as f32;
    let s5 = (2.0 * PI / 5.0).sin() as f32;
    let s10 = (2.0 * PI / 10.0).sin() as f32;
    let c12 = (2.0 * PI / 12.0).cos() as f32;
    [
        c5,
        c5,
        c10,
        c10,
        s5,
        s5,
        s10,
        s10,
        c12,
        c12,
        (2.0 * PI / 6.0).cos() as f32,
        (8.0 * PI / 6.0).cos() as f32,
    ]
}

/// `ff_tx_tab_<n>`: `cos(2 pi i / n)` for `i < n/4`, then 0.
fn tab_sr(n: usize) -> Vec<f32> {
    let freq = 2.0 * PI / n as f64;
    (0..n / 4)
        .map(|i| (i as f64 * freq).cos() as f32)
        .chain(std::iter::once(0.0))
        .collect()
}

/// `fft3` writing `out[at]`, `out[at + stride]`, `out[at + 2 stride]`.
fn fft3(out: &mut [C32], at: usize, stride: usize, input: &[C32], tab: &[f32; 12]) {
    let t0 = input[0];
    let mut t1 = C32 {
        re: input[1].im - input[2].im,
        im: input[1].re - input[2].re,
    };
    let mut t2 = C32 {
        re: input[1].re + input[2].re,
        im: input[1].im + input[2].im,
    };
    out[at] = C32 {
        re: t0.re + t2.re,
        im: t0.im + t2.im,
    };
    t1.re *= tab[8];
    t1.im *= tab[9];
    t2.re *= tab[10];
    t2.im *= tab[10];
    out[at + stride] = C32 {
        re: t0.re - t2.re + t1.re,
        im: t0.im - t2.im - t1.im,
    };
    out[at + 2 * stride] = C32 {
        re: t0.re - t2.re - t1.re,
        im: t0.im - t2.im + t1.im,
    };
}

/// `DECL_FFT5`: a 5-point transform of `input` written to
/// `out[base + d[k] * stride]`.
fn fft5(
    out: &mut [C32],
    base: usize,
    stride: usize,
    d: [usize; 5],
    input: &[C32],
    tab: &[f32; 12],
) {
    let dc = input[0];
    let mut t = [C32::default(); 6];
    (t[1].im, t[0].re) = (input[1].re - input[4].re, input[1].re + input[4].re);
    (t[1].re, t[0].im) = (input[1].im - input[4].im, input[1].im + input[4].im);
    (t[3].im, t[2].re) = (input[2].re - input[3].re, input[2].re + input[3].re);
    (t[3].re, t[2].im) = (input[2].im - input[3].im, input[2].im + input[3].im);

    out[base + d[0] * stride] = C32 {
        re: dc.re + t[0].re + t[2].re,
        im: dc.im + t[0].im + t[2].im,
    };

    (t[4].re, t[0].re) = smul(tab[0], tab[2], t[2].re, t[0].re);
    (t[4].im, t[0].im) = smul(tab[0], tab[2], t[2].im, t[0].im);
    (t[5].re, t[1].re) = cmul(tab[4], tab[6], t[3].re, t[1].re);
    (t[5].im, t[1].im) = cmul(tab[4], tab[6], t[3].im, t[1].im);

    let mut z = [C32::default(); 4];
    (z[0].re, z[3].re) = (t[0].re - t[1].re, t[0].re + t[1].re);
    (z[0].im, z[3].im) = (t[0].im - t[1].im, t[0].im + t[1].im);
    (z[2].re, z[1].re) = (t[4].re - t[5].re, t[4].re + t[5].re);
    (z[2].im, z[1].im) = (t[4].im - t[5].im, t[4].im + t[5].im);

    out[base + d[1] * stride] = C32 {
        re: dc.re + z[3].re,
        im: dc.im + z[0].im,
    };
    out[base + d[2] * stride] = C32 {
        re: dc.re + z[2].re,
        im: dc.im + z[1].im,
    };
    out[base + d[3] * stride] = C32 {
        re: dc.re + z[1].re,
        im: dc.im + z[2].im,
    };
    out[base + d[4] * stride] = C32 {
        re: dc.re + z[0].re,
        im: dc.im + z[3].im,
    };
}

/// `fft15`: three-point transforms, then `fft5_m1/m2/m3`, written to
/// `out[base + k * stride]` in the order the PFA map expects.
fn fft15(out: &mut [C32], base: usize, stride: usize, input: &[C32; 15], tab: &[f32; 12]) {
    let mut tmp = [C32::default(); 15];
    for i in 0..5 {
        fft3(&mut tmp, i, 5, &input[i * 3..i * 3 + 3], tab);
    }
    fft5(out, base, stride, [0, 6, 12, 3, 9], &tmp[0..5], tab);
    fft5(out, base, stride, [10, 1, 7, 13, 4], &tmp[5..10], tab);
    fft5(out, base, stride, [5, 11, 2, 8, 14], &tmp[10..15], tab);
}

/// `BUTTERFLIES(a0, a1, a2, a3)` on `z`, with the `t1, t2, t5, t6` the
/// caller computed.
#[inline]
fn butterflies(
    z: &mut [C32],
    [a0, a1, a2, a3]: [usize; 4],
    t1: f32,
    t2: f32,
    mut t5: f32,
    mut t6: f32,
) {
    let (r0, i0, r1, i1) = (z[a0].re, z[a0].im, z[a1].re, z[a1].im);
    let t3;
    (t3, t5) = (t5 - t1, t5 + t1);
    (z[a2].re, z[a0].re) = (r0 - t5, r0 + t5);
    (z[a3].im, z[a1].im) = (i1 - t3, i1 + t3);
    let t4;
    (t4, t6) = (t2 - t6, t2 + t6);
    (z[a3].re, z[a1].re) = (r1 - t4, r1 + t4);
    (z[a2].im, z[a0].im) = (i0 - t6, i0 + t6);
}

/// `TRANSFORM(a0, a1, a2, a3, wre, wim)` on `z`.
#[inline]
fn transform(z: &mut [C32], a: [usize; 4], wre: f32, wim: f32) {
    let (t1, t2) = cmul(z[a[2]].re, z[a[2]].im, wre, -wim);
    let (t5, t6) = cmul(z[a[3]].re, z[a[3]].im, wre, wim);
    butterflies(z, a, t1, t2, t5, t6);
}

/// The split-radix codelets, in place on `z[..n]`; `tabs` holds
/// `ff_tx_tab_8`, `_16` and `_32`.
struct SplitRadix {
    tab8: Vec<f32>,
    tab16: Vec<f32>,
    tab32: Vec<f32>,
}

impl SplitRadix {
    fn new() -> Self {
        Self {
            tab8: tab_sr(8),
            tab16: tab_sr(16),
            tab32: tab_sr(32),
        }
    }

    /// `ff_tx_fft4_ns`.
    fn fft4(z: &mut [C32]) {
        let (t3, t1) = (z[0].re - z[1].re, z[0].re + z[1].re);
        let (t8, t6) = (z[3].re - z[2].re, z[3].re + z[2].re);
        let (t4, t2) = (z[0].im - z[1].im, z[0].im + z[1].im);
        let (t7, t5) = (z[2].im - z[3].im, z[2].im + z[3].im);
        (z[2].re, z[0].re) = (t1 - t6, t1 + t6);
        (z[3].im, z[1].im) = (t4 - t8, t4 + t8);
        (z[3].re, z[1].re) = (t3 - t7, t3 + t7);
        (z[2].im, z[0].im) = (t2 - t5, t2 + t5);
    }

    /// `ff_tx_fft8_ns`.
    fn fft8(&self, z: &mut [C32]) {
        Self::fft4(&mut z[..4]);
        let (t1, d5re) = (z[4].re - -z[5].re, z[4].re + -z[5].re);
        let (t2, d5im) = (z[4].im - -z[5].im, z[4].im + -z[5].im);
        let (t5, d7re) = (z[6].re - -z[7].re, z[6].re + -z[7].re);
        let (t6, d7im) = (z[6].im - -z[7].im, z[6].im + -z[7].im);
        z[5] = C32 { re: d5re, im: d5im };
        z[7] = C32 { re: d7re, im: d7im };
        butterflies(z, [0, 2, 4, 6], t1, t2, t5, t6);
        let cos = self.tab8[1];
        transform(z, [1, 3, 5, 7], cos, cos);
    }

    /// `ff_tx_fft16_ns`.
    fn fft16(&self, z: &mut [C32]) {
        let (c1, c2, c3) = (self.tab16[1], self.tab16[2], self.tab16[3]);
        self.fft8(&mut z[..8]);
        Self::fft4(&mut z[8..12]);
        Self::fft4(&mut z[12..16]);
        let (t1, t2, t5, t6) = (z[8].re, z[8].im, z[12].re, z[12].im);
        butterflies(z, [0, 4, 8, 12], t1, t2, t5, t6);
        transform(z, [2, 6, 10, 14], c2, c2);
        transform(z, [1, 5, 9, 13], c1, c3);
        transform(z, [3, 7, 11, 15], c3, c1);
    }

    /// `ff_tx_fft32_ns` (`DECL_SR_CODELET(32, 16, 8)`).
    fn fft32(&self, z: &mut [C32]) {
        self.fft16(&mut z[..16]);
        self.fft8(&mut z[16..24]);
        self.fft8(&mut z[24..32]);
        Self::combine(z, &self.tab32, 4);
    }

    /// `ff_tx_fft_sr_combine(z, cos, len)`.
    fn combine(z: &mut [C32], cos: &[f32], len: usize) {
        let (o1, o2, o3) = (2 * len, 4 * len, 6 * len);
        // wim = cos + o1 - 7, walking down by 8 as cos walks up.
        let mut base = 0usize;
        let mut c = 0usize;
        let mut w = o1 - 7;
        let mut i = 0;
        while i < len {
            for (k, wk) in [
                (0, 7),
                (2, 5),
                (4, 3),
                (6, 1),
                (1, 6),
                (3, 4),
                (5, 2),
                (7, 0),
            ] {
                let at = base + k;
                transform(z, [at, at + o1, at + o2, at + o3], cos[c + k], cos[w + wk]);
            }
            base += 8;
            c += 8;
            w = w.wrapping_sub(8);
            i += 4;
        }
    }

    /// The `m`-point codelet, in place.
    fn fft(&self, z: &mut [C32], m: usize) {
        match m {
            4 => Self::fft4(z),
            8 => self.fft8(z),
            16 => self.fft16(z),
            32 => self.fft32(z),
            _ => unreachable!("CELT transforms use M = 4, 8, 16, 32"),
        }
    }
}

/// `C`'s `split_radix_permutation`.
fn split_radix_permutation(i: i32, len: i32, inv: i32) -> i32 {
    let len = len >> 1;
    if len <= 1 {
        return i & 1;
    }
    if i & len == 0 {
        return split_radix_permutation(i, len, inv) * 2;
    }
    let len = len >> 1;
    split_radix_permutation(i, len, inv) * 4 + 1 - 2 * (i32::from(i & len == 0) ^ inv)
}

/// `mulinv(n, m)`: the inverse of `n` modulo `m`.
fn mulinv(n: usize, m: usize) -> usize {
    let n = n % m;
    (1..m).find(|x| (n * x) % m == 1).expect("coprime")
}

/// `av_tx` inverse float MDCT of `len` coefficients
/// (`mdct_pfa_15xM_inv_float_c` with its `fftM_ns_float_c`).
pub(crate) struct Imdct {
    len: usize,
    m: usize,
    /// `s->map`: the doubled input map (`len/2` entries), then the output
    /// map (`len/2` entries).
    map: Vec<usize>,
    /// The sub-transform's scatter map (`s->sub->map`).
    sub_map: Vec<usize>,
    /// Pre-rotation twiddles in input-map order, then the post-rotation
    /// ones (`s->exp`).
    exp: Vec<C32>,
    tab53: [f32; 12],
    sr: SplitRadix,
    /// `s->tmp`, kept between calls.
    tmp: Vec<C32>,
}

impl Imdct {
    /// `ff_tx_mdct_pfa_init` for the 15 x M factorization; `scale` is the
    /// float FFmpeg's caller passes.
    pub(crate) fn new(len: usize, scale: f64) -> Self {
        const N: usize = 15;
        let half = len / 2;
        let m = half / N;

        // ff_tx_gen_compound_mapping(s, NULL, inv = 1, 15, m): gather.
        let mut map = vec![0usize; 2 * half];
        let (m_inv, n_inv) = (mulinv(m, N), mulinv(N, m));
        for j in 0..m {
            for i in 0..N {
                map[j * N + i] = (i * m + j * N) % half;
                map[half + (i * m * m_inv + j * N * n_inv) % half] = i * m + j;
            }
        }
        for i in 0..m {
            let block = &mut map[i * N + 1..i * N + N];
            for j in 0..(N - 1) / 2 {
                block.swap(j, N - j - 2);
            }
        }
        // TX_EMBED_INPUT_PFA_MAP(s->map, len/2, 3, 5).
        for k in (0..half).step_by(N) {
            let mtmp: Vec<usize> = map[k..k + N].to_vec();
            for mm in 0..5 {
                for nn in 0..3 {
                    map[k + mm * 3 + nn] = mtmp[(mm * 3 + nn * 5) % N];
                }
            }
        }

        // ff_tx_mdct_gen_exp(s, s->map): s->len is the MDCT's `len`.
        let len4 = half;
        let scale_d = f64::from(scale as f32);
        let theta = (if scale_d < 0.0 { len4 as f64 } else { 0.0 }) + 1.0 / 8.0;
        let s = scale_d.abs().sqrt();
        let mut exp = vec![C32::default(); 2 * len4];
        for i in 0..len4 {
            let alpha = FRAC_PI_2 * (i as f64 + theta) / len4 as f64;
            exp[len4 + i] = C32 {
                re: (alpha.cos() * s) as f32,
                im: (alpha.sin() * s) as f32,
            };
        }
        for i in 0..len4 {
            exp[i] = exp[len4 + map[i]];
        }
        for v in &mut map[..half] {
            *v <<= 1;
        }

        // The sub-transform: ff_tx_gen_ptwo_revtab, scatter, inv = 1.
        let mut sub_map = vec![0usize; m];
        for i in 0..m {
            let p = split_radix_permutation(i as i32, m as i32, 1);
            sub_map[((-p) & (m as i32 - 1)) as usize] = i;
        }

        Self {
            len,
            m,
            map,
            sub_map,
            exp,
            tab53: tab_53(),
            sr: SplitRadix::new(),
            tmp: vec![C32::default(); half],
        }
    }

    /// `dst[0..len] = imdct(src[0], src[stride], ...)`
    /// (`ff_tx_mdct_pfa_15xM_inv`).
    pub(crate) fn run(&mut self, dst: &mut [f32], src: &[f32], stride: usize) {
        const N: usize = 15;
        let len = self.len;
        let (len2, len4, m) = (len / 2, len / 4, self.m);
        let last = (N * m * 2 - 1) * stride;

        let mut input = [C32::default(); N];
        for g in 0..m {
            for (j, slot) in input.iter_mut().enumerate() {
                let k = self.map[g * N + j];
                let tmp = C32 {
                    re: src[last - k * stride],
                    im: src[k * stride],
                };
                let e = self.exp[g * N + j];
                let (re, im) = cmul(tmp.re, tmp.im, e.re, e.im);
                *slot = C32 { re, im };
            }
            fft15(&mut self.tmp, self.sub_map[g], m, &input, &self.tab53);
        }

        for i in 0..N {
            self.sr.fft(&mut self.tmp[m * i..m * (i + 1)], m);
        }

        let exp = &self.exp[len2..];
        let out_map = &self.map[len2..];
        for i in 0..len4 {
            let (i0, i1) = (len4 + i, len4 - i - 1);
            let (s0, s1) = (out_map[i0], out_map[i1]);
            let src1 = C32 {
                re: self.tmp[s1].im,
                im: self.tmp[s1].re,
            };
            let src0 = C32 {
                re: self.tmp[s0].im,
                im: self.tmp[s0].re,
            };
            let (z1re, z0im) = cmul(src1.re, src1.im, exp[i1].im, exp[i1].re);
            let (z0re, z1im) = cmul(src0.re, src0.im, exp[i0].im, exp[i0].re);
            dst[2 * i1] = z1re;
            dst[2 * i0 + 1] = z0im;
            dst[2 * i0] = z0re;
            dst[2 * i1 + 1] = z1im;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The transform equals `ff_tx_mdct_naive_inv`'s definition, to float
    /// precision.
    #[test]
    fn matches_the_naive_definition() {
        for len in [120usize, 240, 480, 960] {
            let scale = -1.0 / 32768.0;
            let stride = 3;
            let src: Vec<f32> = (0..len * stride)
                .map(|i| ((i * 7919 % 1000) as f32 - 500.0) * 3.0)
                .collect();
            let mut fast = vec![0f32; len];
            Imdct::new(len, scale).run(&mut fast, &src, stride);
            let half = len / 2;
            let phase = PI / (4.0 * len as f64);
            let (mut worst, mut peak) = (0f64, 0f64);
            for i in 0..half {
                let (mut d, mut u) = (0f64, 0f64);
                let id = phase * (2 * len - 2 * i - 1) as f64;
                let iu = phase * (3 * len + 2 * i + 1) as f64;
                for j in 0..len {
                    let a = (2 * j + 1) as f64;
                    let v = f64::from(src[j * stride]);
                    d += (a * id).cos() * v;
                    u += (a * iu).cos() * v;
                }
                peak = peak.max((d * scale).abs()).max((u * scale).abs());
                worst = worst.max((f64::from(fast[i]) - d * scale).abs());
                worst = worst.max((f64::from(fast[i + half]) + u * scale).abs());
            }
            assert!(
                worst < peak * 1e-5,
                "len {len}: max error {worst} of {peak}"
            );
        }
    }
}
