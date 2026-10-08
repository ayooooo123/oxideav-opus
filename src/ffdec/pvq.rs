// Port of FFmpeg's CELT band (de)quantisation, decoder side (FFmpeg commit
// 2da55bf: libavcodec/opus/pvq.c, pvq.h).
// Copyright (c) 2007-2008 CSIRO, (c) 2007-2009 Xiph.Org Foundation,
// (c) 2008-2009 Gregory Maxwell, (c) 2012 Andrew D'Addesio,
// (c) 2013-2014 Mozilla Corporation, (c) 2017 Rostislav Pehlivanov
// <atomnuker@gmail.com>; LGPL-2.1-or-later (see LICENSE-LGPL).

use std::f64::consts::{FRAC_1_SQRT_2, PI};

use super::fp::ordered_dot;
use super::rc::{opus_ilog, RangeDecoder};
use super::tab::*;

pub(crate) const CELT_MAX_BANDS: usize = 21;
const CELT_QTHETA_OFFSET: i32 = 4;
const CELT_QTHETA_OFFSET_TWOPHASE: i32 = 16;
pub(crate) const SPREAD_NONE: u32 = 0;
pub(crate) const SPREAD_AGGRESSIVE: u32 = 3;

/// The per-frame state `quant_band` reads and updates (fields of
/// `CeltFrame` and `CeltPVQ`).
pub(crate) struct PvqState {
    pub(crate) remaining2: i32,
    pub(crate) seed: u32,
    pub(crate) spread: u32,
    pub(crate) intensity_stereo: i32,
    pub(crate) apply_phase_inv: bool,
    pub(crate) tf_change: [i32; CELT_MAX_BANDS],
    qcoeff: [i32; 256],
    hadamard_tmp: [f32; 256],
}

impl PvqState {
    pub(crate) fn new(apply_phase_inv: bool) -> Self {
        Self {
            remaining2: 0,
            seed: 0,
            spread: 0,
            intensity_stereo: 0,
            apply_phase_inv,
            tf_change: [0; CELT_MAX_BANDS],
            qcoeff: [0; 256],
            hadamard_tmp: [0.0; 256],
        }
    }

    /// `celt_rng`.
    pub(crate) fn rng(&mut self) -> u32 {
        self.seed = self
            .seed
            .wrapping_mul(1_664_525)
            .wrapping_add(1_013_904_223);
        self.seed
    }
}

/// Where a folding source (`lowband`) lives.
#[derive(Clone, Copy)]
pub(crate) enum Low {
    /// `norm1`/`norm2`, at this offset of the shared buffer.
    Norm(usize),
    /// `lowband_scratch`, at this offset.
    Scratch(usize),
}

/// The buffers `ff_celt_quant_bands` hands to `quant_band`.
pub(crate) struct Bands<'a> {
    pub(crate) norm: &'a mut [f32],
    pub(crate) scratch: &'a mut [f32],
}

impl Bands<'_> {
    fn low(&self, l: Low, i: usize) -> f32 {
        match l {
            Low::Norm(o) => self.norm.get(o + i).copied().unwrap_or(0.0),
            Low::Scratch(o) => self.scratch.get(o + i).copied().unwrap_or(0.0),
        }
    }

    fn low_mut(&mut self, l: Low) -> &mut [f32] {
        match l {
            Low::Norm(o) => {
                let o = o.min(self.norm.len());
                &mut self.norm[o..]
            }
            Low::Scratch(o) => {
                let o = o.min(self.scratch.len());
                &mut self.scratch[o..]
            }
        }
    }
}

#[inline]
fn mul16(a: i32, b: i32) -> i32 {
    i32::from(a as i16) * i32::from(b as i16)
}

#[inline]
fn round_mul16(a: i32, b: i32) -> i32 {
    (mul16(a, b) + 16384) >> 15
}

/// `celt_cos`.
fn celt_cos(x: i32) -> i32 {
    let mut x = ((mul16(x, x) + 4096) >> 13) as i16 as i32;
    x = ((32767 - x) + round_mul16(x, -7651 + round_mul16(x, 8277 + round_mul16(-626, x)))) as i16
        as i32;
    (x + 1) as i16 as i32
}

/// `celt_log2tan`.
fn celt_log2tan(isin: i32, icos: i32) -> i32 {
    let lc = opus_ilog(icos as u32) as i32;
    let ls = opus_ilog(isin as u32) as i32;
    // Both are in 1..=32767 for any decodable theta; wrap rather than trap.
    let icos = icos.wrapping_shl((15 - lc) as u32);
    let isin = isin.wrapping_shl((15 - ls) as u32);
    (ls << 11) - (lc << 11) + round_mul16(isin, round_mul16(isin, -2597) + 7932)
        - round_mul16(icos, round_mul16(icos, -2597) + 7932)
}

/// `CELT_PVQ_U(n, k)`.
fn pvq_u(n: u32, k: u32) -> u32 {
    let (lo, hi) = (n.min(k) as usize, n.max(k) as usize);
    row(lo, hi)
}

/// `ff_celt_pvq_u_row[r][c]`, reading on through the flat table as C does.
fn row(r: usize, c: usize) -> u32 {
    CELT_PVQ_U_ROW
        .get(r)
        .and_then(|&o| CELT_PVQ_U.get(o + c))
        .copied()
        .unwrap_or(0)
}

/// `CELT_PVQ_V(n, k)`.
fn pvq_v(n: u32, k: u32) -> u32 {
    pvq_u(n, k).wrapping_add(pvq_u(n, k + 1))
}

/// `celt_bits2pulses`.
fn bits2pulses(cache: &[u8], bits: i32) -> i32 {
    let at = |i: i32| i32::from(cache.get(i as usize).copied().unwrap_or(0));
    let mut low = 0i32;
    let mut high = at(0);
    let bits = bits - 1;
    for _ in 0..6 {
        let center = (low + high + 1) >> 1;
        if at(center) >= bits {
            high = center;
        } else {
            low = center;
        }
    }
    let lo_bits = if low == 0 { -1 } else { at(low) };
    if bits - lo_bits <= at(high) - bits {
        low
    } else {
        high
    }
}

/// `celt_pulses2bits`.
fn pulses2bits(cache: &[u8], pulses: i32) -> i32 {
    if pulses == 0 {
        0
    } else {
        i32::from(cache.get(pulses as usize).copied().unwrap_or(0)) + 1
    }
}

fn exp_rotation_impl(x: &mut [f32], len: usize, stride: usize, c: f32, s: f32) {
    if len <= stride {
        return;
    }
    for i in 0..len - stride {
        let x1 = x[i];
        let x2 = x[i + stride];
        // c * x2 + s * x1 and c * x1 - s * x2 fuse their left product
        // (pvq.c:102-103).
        x[i + stride] = c.mul_add(x2, s * x1);
        x[i] = c.mul_add(x1, -(s * x2));
    }
    if len < 2 * stride + 1 {
        return;
    }
    let mut i = len as isize - 2 * stride as isize - 1;
    while i >= 0 {
        let iu = i as usize;
        let x1 = x[iu];
        let x2 = x[iu + stride];
        x[iu + stride] = c.mul_add(x2, s * x1);
        x[iu] = c.mul_add(x1, -(s * x2));
        i -= 1;
    }
}

/// `celt_exp_rotation` (decoder direction).
fn exp_rotation(x: &mut [f32], len: u32, stride: u32, k: u32, spread: u32) {
    if 2 * k >= len || spread == SPREAD_NONE {
        return;
    }
    let gain = len as f32 / (len + (20 - 5 * spread) * k) as f32;
    let theta = (PI * f64::from(gain) * f64::from(gain) / 4.0) as f32;
    let c = theta.cos();
    let s = theta.sin();
    let mut stride2 = 0u32;
    if len >= stride << 3 {
        stride2 = 1;
        while (stride2 * stride2 + stride2) * stride + (stride >> 2) < len {
            stride2 += 1;
        }
    }
    let len = len / stride;
    for i in 0..stride {
        let sub = &mut x[(i * len) as usize..((i + 1) * len) as usize];
        if stride2 != 0 {
            exp_rotation_impl(sub, len as usize, stride2 as usize, s, c);
        }
        exp_rotation_impl(sub, len as usize, 1, c, s);
    }
}

fn extract_collapse_mask(iy: &[i32], n: u32, b: u32) -> u32 {
    if b <= 1 {
        return 1;
    }
    let n0 = (n / b) as usize;
    let mut mask = 0u32;
    for i in 0..b as usize {
        for j in 0..n0 {
            mask |= u32::from(iy[i * n0 + j] != 0) << i;
        }
    }
    mask
}

fn stereo_merge(x: &mut [f32], y: &mut [f32], mid: f32, n: usize) {
    // pvq.c:179: two in-order reductions (contract "Float rounding").
    let xp = ordered_dot(0.0, n, |i| (x[i], y[i])) * mid;
    let side = ordered_dot(0.0, n, |i| (y[i], y[i]));
    let mid2 = mid;
    let e0 = mid2.mul_add(mid2, side) - 2.0 * xp;
    let e1 = mid2.mul_add(mid2, side) + 2.0 * xp;
    if e0 < 6e-4 || e1 < 6e-4 {
        y[..n].copy_from_slice(&x[..n]);
        return;
    }
    let g0 = 1.0 / e0.sqrt();
    let g1 = 1.0 / e1.sqrt();
    for i in 0..n {
        let v0 = mid * x[i];
        let v1 = y[i];
        x[i] = g0 * (v0 - v1);
        y[i] = g1 * (v0 + v1);
    }
}

fn interleave_hadamard(tmp: &mut [f32], x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let order = &CELT_HADAMARD_ORDER[if hadamard { stride - 2 } else { 30 }..];
    for i in 0..stride {
        for j in 0..n0 {
            tmp[j * stride + i] = x[usize::from(order[i]) * n0 + j];
        }
    }
    x[..n].copy_from_slice(&tmp[..n]);
}

fn deinterleave_hadamard(tmp: &mut [f32], x: &mut [f32], n0: usize, stride: usize, hadamard: bool) {
    let n = n0 * stride;
    let order = &CELT_HADAMARD_ORDER[if hadamard { stride - 2 } else { 30 }..];
    for i in 0..stride {
        for j in 0..n0 {
            tmp[usize::from(order[i]) * n0 + j] = x[j * stride + i];
        }
    }
    x[..n].copy_from_slice(&tmp[..n]);
}

fn haar1(x: &mut [f32], n0: usize, stride: usize) {
    let n0 = n0 >> 1;
    for i in 0..stride {
        for j in 0..n0 {
            let a = stride * (2 * j) + i;
            let b = stride * (2 * j + 1) + i;
            let x0 = x[a];
            let x1 = x[b];
            x[a] = (f64::from(x0 + x1) * FRAC_1_SQRT_2) as f32;
            x[b] = (f64::from(x0 - x1) * FRAC_1_SQRT_2) as f32;
        }
    }
}

fn compute_qn(n: i32, b: i32, offset: i32, pulse_cap: i32, stereo: bool) -> i32 {
    let mut n2 = 2 * n - 1;
    if stereo && n == 2 {
        n2 -= 1;
    }
    let qb = (b - pulse_cap - (4 << 3))
        .min((b + n2 * offset) / n2)
        .min(8 << 3);
    if qb < (1 << 3 >> 1) {
        1
    } else {
        ((i32::from(CELT_QN_EXP2[(qb & 7) as usize]) >> (14 - (qb >> 3))) + 1) >> 1 << 1
    }
}

/// `celt_cwrsi`: the pulse vector of index `i`; returns its squared norm.
fn cwrsi(n: u32, k: u32, i: u32, y: &mut [i32]) -> u64 {
    let (mut n, mut k, mut i) = (n, k, i);
    let mut norm = 0u64;
    let mut yi = 0usize;
    let mut put = |y: &mut [i32], v: i32| {
        if let Some(slot) = y.get_mut(yi) {
            *slot = v;
        }
        yi += 1;
    };
    while n > 2 {
        if k >= n {
            let r = n as usize;
            let p = row(r, k as usize + 1);
            let s: i32 = -i32::from(i >= p);
            i = i.wrapping_sub(p & s as u32);
            let k0 = k;
            let q = row(r, n as usize);
            let mut p;
            if q > i {
                k = n;
                loop {
                    k -= 1;
                    p = row(k as usize, n as usize);
                    if p <= i || k == 0 {
                        break;
                    }
                }
            } else {
                p = row(r, k as usize);
                while p > i && k > 0 {
                    k -= 1;
                    p = row(r, k as usize);
                }
            }
            i = i.wrapping_sub(p);
            let val = ((k0.wrapping_sub(k) as i32).wrapping_add(s)) ^ s;
            norm += (i64::from(val) * i64::from(val)) as u64;
            put(y, val);
        } else {
            let p = row(k as usize, n as usize);
            let q = row(k as usize + 1, n as usize);
            if p <= i && i < q {
                i -= p;
                put(y, 0);
            } else {
                let s: i32 = -i32::from(i >= q);
                i = i.wrapping_sub(q & s as u32);
                let k0 = k;
                let mut p;
                loop {
                    if k == 0 {
                        p = 0;
                        break;
                    }
                    k -= 1;
                    p = row(k as usize, n as usize);
                    if p <= i {
                        break;
                    }
                }
                i = i.wrapping_sub(p);
                let val = ((k0.wrapping_sub(k) as i32).wrapping_add(s)) ^ s;
                norm += (i64::from(val) * i64::from(val)) as u64;
                put(y, val);
            }
        }
        n -= 1;
    }
    // n == 2
    let p = 2 * k + 1;
    let s: i32 = -i32::from(i >= p);
    i = i.wrapping_sub(p & s as u32);
    let k0 = k;
    k = (i.wrapping_add(1)) / 2;
    if k != 0 {
        i = i.wrapping_sub(2 * k - 1);
    }
    let val = ((k0.wrapping_sub(k) as i32).wrapping_add(s)) ^ s;
    norm += (i64::from(val) * i64::from(val)) as u64;
    put(y, val);
    // n == 1
    let s: i32 = (i as i32).wrapping_neg();
    let val = ((k as i32).wrapping_add(s)) ^ s;
    norm += (i64::from(val) * i64::from(val)) as u64;
    put(y, val);
    norm
}

/// `celt_alg_unquant`.
fn alg_unquant(
    st: &mut PvqState,
    rc: &mut RangeDecoder,
    x: &mut [f32],
    n: u32,
    k: u32,
    blocks: u32,
    gain: f32,
) -> u32 {
    let idx = rc.dec_uint(pvq_v(n, k));
    let norm = cwrsi(n, k, idx, &mut st.qcoeff);
    let gain = gain / (norm as f32).sqrt();
    for i in 0..n as usize {
        x[i] = gain * st.qcoeff[i] as f32;
    }
    exp_rotation(x, n, blocks, k, st.spread);
    extract_collapse_mask(&st.qcoeff, n, blocks)
}

fn renormalize(x: &mut [f32], n: usize, gain: f32) {
    // celt.h:161: an in-order reduction.
    let g = ordered_dot(1e-15, n, |k| (x[k], x[k]));
    let g = gain / g.sqrt();
    for v in &mut x[..n] {
        *v *= g;
    }
}

/// What the split part of `quant_band_template` hands back.
struct Split {
    cm: u32,
    mid: f32,
    inv: bool,
}

/// `quant_band_template` with `quant` = 0 (`pvq_decode_band`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn quant_band(
    st: &mut PvqState,
    rc: &mut RangeDecoder,
    bufs: &mut Bands,
    band: usize,
    x: &mut [f32],
    mut y: Option<&mut [f32]>,
    n: i32,
    b: i32,
    mut blocks: u32,
    mut lowband: Option<Low>,
    mut duration: i32,
    lowband_out: Option<usize>,
    level: i32,
    gain: f32,
    lowband_scratch: Option<usize>,
    mut fill: u32,
) -> u32 {
    let stereo = y.is_some();
    let mut split = stereo;
    let n0 = n as u32;
    let mut n = n;
    let mut n_b = n / blocks as i32;
    let mut n_b0 = n_b;
    let mut b0 = blocks;
    let mut time_divide = 0;
    let mut recombine = 0;
    let mut inv = false;
    let mut mid = 0f32;
    let longblocks = b0 == 1;
    let mut cm: u32;

    if n == 1 {
        for i in 0..=usize::from(stereo) {
            let mut sign = 0u32;
            if st.remaining2 >= 1 << 3 {
                sign = rc.get_raw(1);
                st.remaining2 -= 1 << 3;
            }
            let v = 1.0f32 - 2.0 * sign as f32;
            if i == 0 {
                x[0] = v;
            } else if let Some(y) = y.as_deref_mut() {
                y[0] = v;
            }
        }
        if let Some(o) = lowband_out {
            bufs.norm[o] = x[0];
        }
        return 1;
    }

    if !stereo && level == 0 {
        let mut tf_change = st.tf_change[band];
        if tf_change > 0 {
            recombine = tf_change;
        }
        if let (Some(lb), Some(s)) = (lowband, lowband_scratch) {
            if recombine != 0 || ((n_b & 1) == 0 && tf_change < 0) || b0 > 1 {
                for i in 0..n as usize {
                    let v = bufs.low(lb, i);
                    bufs.scratch[s + i] = v;
                }
                lowband = Some(Low::Scratch(s));
            }
        }
        for k in 0..recombine {
            if let Some(lb) = lowband {
                haar1(bufs.low_mut(lb), (n >> k) as usize, 1 << k);
            }
            fill = u32::from(CELT_BIT_INTERLEAVE[(fill & 0xf) as usize])
                | u32::from(CELT_BIT_INTERLEAVE[((fill >> 4) & 0xf) as usize]) << 2;
        }
        blocks >>= recombine;
        n_b <<= recombine;

        while (n_b & 1) == 0 && tf_change < 0 {
            if let Some(lb) = lowband {
                haar1(bufs.low_mut(lb), n_b as usize, blocks as usize);
            }
            fill |= fill << blocks;
            blocks <<= 1;
            n_b >>= 1;
            time_divide += 1;
            tf_change += 1;
        }
        b0 = blocks;
        n_b0 = n_b;

        if b0 > 1 {
            if let Some(lb) = lowband {
                let tmp = &mut st.hadamard_tmp;
                deinterleave_hadamard(
                    tmp,
                    bufs.low_mut(lb),
                    (n_b >> recombine) as usize,
                    (b0 << recombine) as usize,
                    longblocks,
                );
            }
        }
    }

    let cache_index = CELT_CACHE_INDEX
        .get(((duration + 1) * CELT_MAX_BANDS as i32 + band as i32).max(0) as usize)
        .copied()
        .unwrap_or(0)
        .max(0) as usize;
    let cache = &CELT_CACHE_BITS[cache_index.min(CELT_CACHE_BITS.len())..];
    let cache0 = i32::from(cache.first().copied().unwrap_or(0));
    let cache_max = i32::from(cache.get(cache0 as usize).copied().unwrap_or(0));

    if !stereo && duration >= 0 && b > cache_max + 12 && n > 2 {
        // Y = X + N: the band splits in two halves.
        n >>= 1;
        split = true;
        duration -= 1;
        if blocks == 1 {
            fill = (fill & 1) | (fill << 1);
        }
        blocks = (blocks + 1) >> 1;
    }

    if split {
        let s = if let Some(yv) = y.as_deref_mut() {
            split_band(
                st,
                rc,
                bufs,
                band,
                x,
                yv,
                true,
                n,
                b,
                blocks,
                b0,
                lowband,
                duration,
                lowband_out,
                level,
                gain,
                lowband_scratch,
                fill,
            )
        } else {
            let (xa, yb) = x.split_at_mut(n as usize);
            split_band(
                st,
                rc,
                bufs,
                band,
                xa,
                yb,
                false,
                n,
                b,
                blocks,
                b0,
                lowband,
                duration,
                lowband_out,
                level,
                gain,
                lowband_scratch,
                fill,
            )
        };
        cm = s.cm;
        mid = s.mid;
        inv = s.inv;
    } else {
        let mut q = bits2pulses(cache, b);
        let mut curr_bits = pulses2bits(cache, q);
        st.remaining2 -= curr_bits;
        while st.remaining2 < 0 && q > 0 {
            st.remaining2 += curr_bits;
            q -= 1;
            curr_bits = pulses2bits(cache, q);
            st.remaining2 -= curr_bits;
        }
        if q != 0 {
            let k = if q < 8 {
                q
            } else {
                (8 + (q & 7)) << ((q >> 3) - 1)
            };
            cm = alg_unquant(st, rc, x, n as u32, k as u32, blocks, gain);
        } else {
            let cm_mask = mask(blocks);
            fill &= cm_mask;
            if fill != 0 {
                match lowband {
                    None => {
                        for v in x.iter_mut().take(n as usize) {
                            *v = ((st.rng() as i32) >> 20) as f32;
                        }
                        cm = cm_mask;
                    }
                    Some(lb) => {
                        for i in 0..n as usize {
                            let noise = if st.rng() & 0x8000 != 0 {
                                1.0f32 / 256.0
                            } else {
                                -1.0f32 / 256.0
                            };
                            x[i] = bufs.low(lb, i) + noise;
                        }
                        cm = fill;
                    }
                }
                renormalize(x, n as usize, gain);
            } else {
                x[..n as usize].fill(0.0);
                cm = 0;
            }
        }
    }

    if stereo {
        let yv = y.expect("stereo band");
        if n > 2 {
            stereo_merge(x, yv, mid, n as usize);
        }
        if inv {
            for v in yv.iter_mut().take(n as usize) {
                *v = -*v;
            }
        }
    } else if level == 0 {
        if b0 > 1 {
            let tmp = &mut st.hadamard_tmp;
            interleave_hadamard(
                tmp,
                x,
                (n_b >> recombine) as usize,
                (b0 << recombine) as usize,
                longblocks,
            );
        }
        n_b = n_b0;
        blocks = b0;
        for _ in 0..time_divide {
            blocks >>= 1;
            n_b <<= 1;
            cm |= cm >> blocks;
            haar1(x, n_b as usize, blocks as usize);
        }
        for k in 0..recombine {
            cm = u32::from(CELT_BIT_DEINTERLEAVE.get(cm as usize).copied().unwrap_or(0));
            haar1(x, (n0 >> k) as usize, 1 << k);
        }
        blocks <<= recombine;
        if let Some(o) = lowband_out {
            let nn = (n0 as f32).sqrt();
            for i in 0..n0 as usize {
                bufs.norm[o + i] = nn * x[i];
            }
        }
        cm &= mask(blocks);
    }
    cm
}

/// `(1 << blocks) - 1`.
fn mask(blocks: u32) -> u32 {
    if blocks >= 32 {
        u32::MAX
    } else {
        (1u32 << blocks) - 1
    }
}

/// The `if (split)` part of `quant_band_template`: `xs` and `ys` are the
/// two halves of a mono band, or the two channels of a stereo one.
#[allow(clippy::too_many_arguments)]
fn split_band(
    st: &mut PvqState,
    rc: &mut RangeDecoder,
    bufs: &mut Bands,
    band: usize,
    xs: &mut [f32],
    ys: &mut [f32],
    stereo: bool,
    n: i32,
    mut b: i32,
    blocks: u32,
    b0: u32,
    lowband: Option<Low>,
    duration: i32,
    lowband_out: Option<usize>,
    level: i32,
    gain: f32,
    lowband_scratch: Option<usize>,
    mut fill: u32,
) -> Split {
    let mut itheta = 0i32;
    let mut inv = false;
    let cm;
    let pulse_cap = i32::from(CELT_LOG_FREQ_RANGE[band]) + duration * 8;
    let offset = (pulse_cap >> 1)
        - if stereo && n == 2 {
            CELT_QTHETA_OFFSET_TWOPHASE
        } else {
            CELT_QTHETA_OFFSET
        };
    let qn = if stereo && band as i32 >= st.intensity_stereo {
        1
    } else {
        compute_qn(n, b, offset, pulse_cap, stereo)
    };
    let tell = rc.tell_frac() as i32;
    if qn != 1 {
        if stereo && n > 2 {
            itheta = rc.dec_uint_step((qn / 2) as u32) as i32;
        } else if stereo || b0 > 1 {
            itheta = rc.dec_uint((qn + 1) as u32) as i32;
        } else {
            itheta = rc.dec_uint_tri(qn as u32) as i32;
        }
        itheta = itheta * 16384 / qn;
    } else if stereo {
        inv = if b > 2 << 3 && st.remaining2 > 2 << 3 {
            rc.dec_log(2) != 0
        } else {
            false
        };
        inv = st.apply_phase_inv && inv;
        itheta = 0;
    }
    let qalloc = rc.tell_frac() as i32 - tell;
    b -= qalloc;

    let orig_fill = fill;
    let (imid, iside, mut delta);
    if itheta == 0 {
        imid = 32767;
        iside = 0;
        fill &= mask(blocks);
        delta = -16384;
    } else if itheta == 16384 {
        imid = 0;
        iside = 32767;
        fill &= mask(blocks).wrapping_shl(blocks);
        delta = 16384;
    } else {
        imid = celt_cos(itheta);
        iside = celt_cos(16384 - itheta);
        delta = round_mul16((n - 1) << 7, celt_log2tan(iside, imid));
    }
    let mid = imid as f32 / 32768.0;
    let side = iside as f32 / 32768.0;

    if n == 2 && stereo {
        let mut mbits = b;
        let sbits = if itheta != 0 && itheta != 16384 {
            1 << 3
        } else {
            0
        };
        mbits -= sbits;
        let c = itheta > 8192;
        st.remaining2 -= qalloc + sbits;
        let mut sign = 0u32;
        if sbits != 0 {
            sign = rc.get_raw(1);
        }
        let sign = 1.0f32 - 2.0 * sign as f32;
        // x2 = c ? Y : X; y2 = c ? X : Y.
        if c {
            cm = quant_band(
                st,
                rc,
                bufs,
                band,
                ys,
                None,
                n,
                mbits,
                blocks,
                lowband,
                duration,
                lowband_out,
                level,
                gain,
                lowband_scratch,
                orig_fill,
            );
            let (x20, x21) = (ys[0], ys[1]);
            xs[0] = -sign * x21;
            xs[1] = sign * x20;
        } else {
            cm = quant_band(
                st,
                rc,
                bufs,
                band,
                xs,
                None,
                n,
                mbits,
                blocks,
                lowband,
                duration,
                lowband_out,
                level,
                gain,
                lowband_scratch,
                orig_fill,
            );
            let (x20, x21) = (xs[0], xs[1]);
            ys[0] = -sign * x21;
            ys[1] = sign * x20;
        }
        xs[0] *= mid;
        xs[1] *= mid;
        ys[0] *= side;
        ys[1] *= side;
        let tmp = xs[0];
        xs[0] = tmp - ys[0];
        ys[0] += tmp;
        let tmp = xs[1];
        xs[1] = tmp - ys[1];
        ys[1] += tmp;
        return Split { cm, mid, inv };
    }

    let mut next_lowband2 = None;
    let mut next_lowband_out1 = None;
    let mut next_level = 0;
    if b0 > 1 && !stereo && (itheta & 0x3fff) != 0 {
        if itheta > 8192 {
            delta -= delta >> (4 - duration);
        } else {
            delta = 0.min(delta + (n << 3 >> (5 - duration)));
        }
    }
    // av_clip((b - delta) / 2, 0, b): below 0 gives 0, even when b < 0.
    let half = (b - delta) / 2;
    let mut mbits = if half < 0 {
        0
    } else if half > b {
        b
    } else {
        half
    };
    let mut sbits = b - mbits;
    st.remaining2 -= qalloc;

    if let Some(lb) = lowband {
        if !stereo {
            next_lowband2 = Some(match lb {
                Low::Norm(o) => Low::Norm(o + n as usize),
                Low::Scratch(o) => Low::Scratch(o + n as usize),
            });
        }
    }
    if stereo {
        next_lowband_out1 = lowband_out;
    } else {
        next_level = level + 1;
    }
    let mut rebalance = st.remaining2;
    let side_shift = (b0 >> 1) & u32::from(stereo).wrapping_sub(1);
    let mid_gain = if stereo { 1.0 } else { gain * mid };
    let mut c;
    if mbits >= sbits {
        c = quant_band(
            st,
            rc,
            bufs,
            band,
            xs,
            None,
            n,
            mbits,
            blocks,
            lowband,
            duration,
            next_lowband_out1,
            next_level,
            mid_gain,
            lowband_scratch,
            fill,
        );
        rebalance = mbits - (rebalance - st.remaining2);
        if rebalance > 3 << 3 && itheta != 0 {
            sbits += rebalance - (3 << 3);
        }
        let cmt = quant_band(
            st,
            rc,
            bufs,
            band,
            ys,
            None,
            n,
            sbits,
            blocks,
            next_lowband2,
            duration,
            None,
            next_level,
            gain * side,
            None,
            fill >> blocks,
        );
        c |= cmt << side_shift;
    } else {
        c = quant_band(
            st,
            rc,
            bufs,
            band,
            ys,
            None,
            n,
            sbits,
            blocks,
            next_lowband2,
            duration,
            None,
            next_level,
            gain * side,
            None,
            fill >> blocks,
        );
        c <<= side_shift;
        rebalance = sbits - (rebalance - st.remaining2);
        if rebalance > 3 << 3 && itheta != 16384 {
            mbits += rebalance - (3 << 3);
        }
        c |= quant_band(
            st,
            rc,
            bufs,
            band,
            xs,
            None,
            n,
            mbits,
            blocks,
            lowband,
            duration,
            next_lowband_out1,
            next_level,
            mid_gain,
            lowband_scratch,
            fill,
        );
    }
    Split { cm: c, mid, inv }
}
