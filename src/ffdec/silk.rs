// Port of FFmpeg's Opus SILK decoder (FFmpeg commit 2da55bf:
// libavcodec/opus/silk.c, silk.h).
// Copyright (c) 2012 Andrew D'Addesio, (c) 2013-2014 Mozilla Corporation;
// LGPL-2.1-or-later (see LICENSE-LGPL).

use super::parse::{BANDWIDTH_NARROWBAND, BANDWIDTH_WIDEBAND};
use super::rc::{opus_ilog, RangeDecoder};
use super::tab::*;

const SILK_HISTORY: usize = 322;
const LTP_ORDER: usize = 5;
/// Maximum residual history (4.2.7.6.1).
const SILK_MAX_LAG: usize = 288 + LTP_ORDER / 2;

#[inline]
fn mulh(a: i32, b: i32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> 32) as i32
}

#[inline]
fn mull(a: i32, b: i32, s: u32) -> i32 {
    ((i64::from(a) * i64::from(b)) >> s) as i32
}

/// `ROUND_MULL(a, b, s)`, 64-bit.
#[inline]
fn round_mull(a: i64, b: i64, s: u32) -> i64 {
    (((a * b) >> (s - 1)) + 1) >> 1
}

#[inline]
fn clipf(v: f32, lo: f32, hi: f32) -> f32 {
    if v < lo {
        lo
    } else if v > hi {
        hi
    } else {
        v
    }
}

/// `sum + Σ a_k b_k` for `k < n`, `term(k) = (a_k, b_k)`, rounded the way
/// FFmpeg's arm64 build evaluates silk.c's LPC loops (`sum -= c * x`,
/// `sum += c * x`): clang vectorizes the in-order reduction in blocks of four
/// products, each rounded and then added in `k` order, and leaves the last
/// `n % 4` terms to fused multiply-adds (LPC order 16: all blocked; order
/// 10: eight blocked, two fused). SILK's synthesis filters can amplify a
/// last-bit difference, so the rounding order matters.
#[inline]
fn ordered_dot(mut sum: f32, n: usize, term: impl Fn(usize) -> (f32, f32)) -> f32 {
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

#[derive(Clone)]
struct SilkFrame {
    coded: bool,
    log_gain: i32,
    nlsf: [i16; 16],
    lpc: [f32; 16],
    output: [f32; 2 * SILK_HISTORY],
    lpc_history: [f32; 2 * SILK_HISTORY],
    primarylag: i32,
    prev_voiced: bool,
}

impl Default for SilkFrame {
    fn default() -> Self {
        Self {
            coded: false,
            log_gain: 0,
            nlsf: [0; 16],
            lpc: [0.0; 16],
            output: [0.0; 2 * SILK_HISTORY],
            lpc_history: [0.0; 2 * SILK_HISTORY],
            primarylag: 0,
            prev_voiced: false,
        }
    }
}

impl SilkFrame {
    /// `silk_flush_frame`.
    fn flush(&mut self) {
        if !self.coded {
            return;
        }
        *self = SilkFrame::default();
    }
}

/// `SilkContext`.
pub(crate) struct Silk {
    output_channels: usize,
    midonly: bool,
    subframes: usize,
    sflength: usize,
    flength: usize,
    nlsf_interp_factor: i32,
    bandwidth: usize,
    wb: bool,
    frame: [SilkFrame; 2],
    prev_stereo_weights: [f32; 2],
    stereo_weights: [f32; 2],
    prev_coded_channels: usize,
}

/// `silk_stabilize_lsf`.
fn stabilize_lsf(nlsf: &mut [i16; 16], order: usize, min_delta: &[u16]) {
    for _pass in 0..20 {
        let mut k = 0usize;
        let mut min_diff = 0i32;
        for i in 0..=order {
            let low = if i != 0 { i32::from(nlsf[i - 1]) } else { 0 };
            let high = if i != order {
                i32::from(nlsf[i])
            } else {
                32768
            };
            let diff = (high - low) - i32::from(min_delta[i]);
            if diff < min_diff {
                min_diff = diff;
                k = i;
            }
        }
        if min_diff == 0 {
            return;
        }
        if k == 0 {
            nlsf[0] = min_delta[0] as i16;
        } else if k == order {
            nlsf[order - 1] = (32768 - i32::from(min_delta[order])) as i16;
        } else {
            let mut min_center = 0i32;
            let mut max_center = 32768i32;
            for &d in &min_delta[..k] {
                min_center += i32::from(d);
            }
            min_center += i32::from(min_delta[k]) >> 1;
            for i in (k + 1..=order).rev() {
                max_center -= i32::from(min_delta[i]);
            }
            max_center -= i32::from(min_delta[k]) >> 1;
            let mut center_val = i32::from(nlsf[k - 1]) + i32::from(nlsf[k]);
            center_val = (center_val >> 1) + (center_val & 1);
            center_val = max_center.min(min_center.max(center_val));
            nlsf[k - 1] = (center_val - (i32::from(min_delta[k]) >> 1)) as i16;
            nlsf[k] = (i32::from(nlsf[k - 1]) + i32::from(min_delta[k])) as i16;
        }
    }

    // The fall-back: sort, then push the LSFs apart.
    for i in 1..order {
        let value = nlsf[i];
        let mut j = i as isize - 1;
        while j >= 0 && nlsf[j as usize] > value {
            nlsf[(j + 1) as usize] = nlsf[j as usize];
            j -= 1;
        }
        nlsf[(j + 1) as usize] = value;
    }
    if i32::from(nlsf[0]) < i32::from(min_delta[0]) {
        nlsf[0] = min_delta[0] as i16;
    }
    for i in 1..order {
        let v = (i32::from(nlsf[i - 1]) + i32::from(min_delta[i])).min(32767);
        nlsf[i] = i32::from(nlsf[i]).max(v) as i16;
    }
    if i32::from(nlsf[order - 1]) > 32768 - i32::from(min_delta[order]) {
        nlsf[order - 1] = (32768 - i32::from(min_delta[order])) as i16;
    }
    for i in (0..order - 1).rev() {
        if i32::from(nlsf[i]) > i32::from(nlsf[i + 1]) - i32::from(min_delta[i + 1]) {
            nlsf[i] = (i32::from(nlsf[i + 1]) - i32::from(min_delta[i + 1])) as i16;
        }
    }
}

/// `silk_is_lpc_stable`.
fn is_lpc_stable(lpc: &[i16; 16], order: usize) -> bool {
    let mut lpc32 = [[0i32; 16]; 2];
    let mut dc_resp = 0i32;
    let mut totalinvgain: i32 = 1 << 30;
    let mut cur = 0usize;
    for k in 0..order {
        dc_resp += i32::from(lpc[k]);
        lpc32[0][k] = i32::from(lpc[k]) * 4096;
    }
    if dc_resp >= 4096 {
        return false;
    }
    let mut k = order - 1;
    loop {
        let row_k = lpc32[cur][k];
        if row_k.unsigned_abs() > 16_773_022 {
            return false;
        }
        let rc = row_k.wrapping_mul(128).wrapping_neg();
        let gaindiv = (1i32 << 30).wrapping_sub(mulh(rc, rc));
        totalinvgain = mulh(totalinvgain, gaindiv).wrapping_shl(2);
        if k == 0 {
            return totalinvgain >= 107_374;
        }
        let fbits = opus_ilog(gaindiv as u32);
        let mut gain = ((1i32 << 29) - 1) / (gaindiv >> (fbits + 1 - 16));
        let error =
            (1i32 << 29).wrapping_sub(mull(gaindiv.wrapping_shl(15 + 16 - fbits), gain, 16));
        gain = gain
            .wrapping_shl(16)
            .wrapping_add(((i64::from(error) * i64::from(gain)) >> 13) as i32);

        let prev = cur;
        cur = k & 1;
        let prevrow = lpc32[prev];
        for j in 0..k {
            let x = prevrow[j].saturating_sub(round_mull(
                i64::from(prevrow[k - j - 1]),
                i64::from(rc),
                31,
            ) as i32);
            let tmp = round_mull(i64::from(x), i64::from(gain), fbits);
            if tmp < i64::from(i32::MIN) || tmp > i64::from(i32::MAX) {
                return false;
            }
            lpc32[cur][j] = tmp as i32;
        }
        k -= 1;
    }
}

/// `silk_lsp2poly`.
fn lsp2poly(lsp: &[i32], pol: &mut [i32; 9], half_order: usize) {
    pol[0] = 65536;
    pol[1] = -lsp[0];
    for i in 1..half_order {
        pol[i + 1] = pol[i - 1].wrapping_mul(2).wrapping_sub(round_mull(
            i64::from(lsp[2 * i]),
            i64::from(pol[i]),
            16,
        ) as i32);
        for j in (2..=i).rev() {
            pol[j] = pol[j].wrapping_add(pol[j - 2]).wrapping_sub(round_mull(
                i64::from(lsp[2 * i]),
                i64::from(pol[j - 1]),
                16,
            ) as i32);
        }
        pol[1] = pol[1].wrapping_sub(lsp[2 * i]);
    }
}

/// `silk_lsf2lpc`.
fn lsf2lpc(nlsf: &[i16; 16], lpcf: &mut [f32; 16], order: usize) {
    let mut lsp = [0i32; 16];
    let mut p = [0i32; 9];
    let mut q = [0i32; 9];
    let mut lpc32 = [0i32; 16];
    let mut lpc = [0i16; 16];

    for k in 0..order {
        let index = (nlsf[k] >> 8) as usize;
        let offset = i32::from(nlsf[k] & 255);
        let k2 = if order == 10 {
            usize::from(SILK_LSF_ORDERING_NBMB[k])
        } else {
            usize::from(SILK_LSF_ORDERING_WB[k])
        };
        let c0 = i32::from(SILK_COSINE[index]);
        let c1 = i32::from(SILK_COSINE[index + 1]);
        let mut v = c0 * 256;
        v += (c1 - c0) * offset;
        lsp[k2] = (v + 4) >> 3;
    }

    lsp2poly(&lsp, &mut p, order >> 1);
    lsp2poly(&lsp[1..], &mut q, order >> 1);

    for k in 0..order >> 1 {
        let p_tmp = p[k + 1].wrapping_add(p[k]);
        let q_tmp = q[k + 1].wrapping_sub(q[k]);
        lpc32[k] = q_tmp.wrapping_neg().wrapping_sub(p_tmp);
        lpc32[order - k - 1] = q_tmp.wrapping_sub(p_tmp);
    }

    let mut i = 0;
    while i < 10 {
        let mut maxabs: u32 = 0;
        let mut k = 0usize;
        for j in 0..order {
            // FFmpeg reads lpc32[k], not lpc32[j]: only the first
            // coefficient is ever measured. Kept, to decode as FFmpeg does.
            let x = lpc32[k].unsigned_abs();
            if x > maxabs {
                maxabs = x;
                k = j;
            }
        }
        maxabs = (maxabs.wrapping_add(16)) >> 5;
        if maxabs > 32767 {
            maxabs = maxabs.min(163_838);
            let chirp_base =
                65470u32.wrapping_sub(((maxabs - 32767) << 14) / ((maxabs * (k as u32 + 1)) >> 2));
            let mut chirp = chirp_base;
            for v in lpc32.iter_mut().take(order) {
                *v = round_mull(i64::from(*v), i64::from(chirp), 16) as i32;
                chirp = ((u64::from(chirp_base) * u64::from(chirp) + 32768) >> 16) as u32;
            }
        } else {
            break;
        }
        i += 1;
    }

    if i == 10 {
        for k in 0..order {
            let x = (lpc32[k].wrapping_add(16)) >> 5;
            lpc[k] = x.clamp(-32768, 32767) as i16;
            lpc32[k] = i32::from(lpc[k]) << 5;
        }
    } else {
        for k in 0..order {
            lpc[k] = ((lpc32[k].wrapping_add(16)) >> 5) as i16;
        }
    }

    let mut i = 1;
    while i <= 16 && !is_lpc_stable(&lpc, order) {
        let chirp_base: u32 = 65536 - (1 << i);
        let mut chirp = chirp_base;
        for k in 0..order {
            lpc32[k] = round_mull(i64::from(lpc32[k]), i64::from(chirp), 16) as i32;
            lpc[k] = ((lpc32[k].wrapping_add(16)) >> 5) as i16;
            chirp = ((u64::from(chirp_base) * u64::from(chirp) + 32768) >> 16) as u32;
        }
        i += 1;
    }

    for k in 0..order {
        lpcf[k] = f32::from(lpc[k]) / 4096.0;
    }
}

/// `silk_count_children`.
fn count_children(rc: &mut RangeDecoder, model: usize, total: i32, child: &mut [i32]) {
    if total != 0 {
        let off = (((total - 1 + 5) * (total - 1)) >> 1) as usize;
        child[0] = rc.dec_cdf(&SILK_MODEL_PULSE_LOCATION[model][off..]) as i32;
        child[1] = total - child[0];
    } else {
        child[0] = 0;
        child[1] = 0;
    }
}

#[derive(Clone, Copy, Default)]
struct Subframe {
    gain: f32,
    pitchlag: i32,
    ltptaps: [f32; 5],
}

impl Silk {
    /// `ff_silk_init` (plus the `ff_silk_flush` it ends with).
    pub(crate) fn new(output_channels: usize) -> Self {
        Self {
            output_channels,
            midonly: false,
            subframes: 0,
            sflength: 0,
            flength: 0,
            nlsf_interp_factor: 0,
            bandwidth: 0,
            wb: false,
            frame: [SilkFrame::default(), SilkFrame::default()],
            prev_stereo_weights: [0.0; 2],
            stereo_weights: [0.0; 2],
            prev_coded_channels: 0,
        }
    }

    /// `ff_silk_flush`.
    pub(crate) fn flush(&mut self) {
        self.frame[0].flush();
        self.frame[1].flush();
        self.prev_stereo_weights = [0.0; 2];
    }

    /// `silk_decode_lpc`.
    fn decode_lpc(
        &mut self,
        channel: usize,
        rc: &mut RangeDecoder,
        lpc_leadin: &mut [f32; 16],
        lpc: &mut [f32; 16],
        voiced: usize,
    ) -> (usize, bool) {
        let wb = self.wb;
        let order = if wb { 16 } else { 10 };
        let mut lsf_i2 = [0i32; 16];
        let mut lsf_res = [0i32; 16];
        let mut nlsf = [0i16; 16];

        let lsf_i1 = rc.dec_cdf(&SILK_MODEL_LSF_S1[usize::from(wb)][voiced]) as usize;
        for i in 0..order {
            let index = if wb {
                usize::from(SILK_LSF_S2_MODEL_SEL_WB[lsf_i1][i])
            } else {
                usize::from(SILK_LSF_S2_MODEL_SEL_NBMB[lsf_i1][i])
            };
            lsf_i2[i] = rc.dec_cdf(&SILK_MODEL_LSF_S2[index]) as i32 - 4;
            if lsf_i2[i] == -4 {
                lsf_i2[i] -= rc.dec_cdf(&SILK_MODEL_LSF_S2_EXT) as i32;
            } else if lsf_i2[i] == 4 {
                lsf_i2[i] += rc.dec_cdf(&SILK_MODEL_LSF_S2_EXT) as i32;
            }
        }

        for i in (0..order).rev() {
            let qstep = if wb { 9830 } else { 11796 };
            // int16_t arithmetic, as in C.
            let mut r = (lsf_i2[i] * 1024) as i16 as i32;
            if lsf_i2[i] < 0 {
                r = (r + 102) as i16 as i32;
            } else if lsf_i2[i] > 0 {
                r = (r - 102) as i16 as i32;
            }
            r = ((r * qstep) >> 16) as i16 as i32;
            if i + 1 < order {
                let weight = if wb {
                    i32::from(
                        SILK_LSF_PRED_WEIGHTS_WB[usize::from(SILK_LSF_WEIGHT_SEL_WB[lsf_i1][i])][i],
                    )
                } else {
                    i32::from(
                        SILK_LSF_PRED_WEIGHTS_NBMB
                            [usize::from(SILK_LSF_WEIGHT_SEL_NBMB[lsf_i1][i])][i],
                    )
                };
                r = (r + ((lsf_res[i + 1] * weight) >> 8)) as i16 as i32;
            }
            lsf_res[i] = r;
        }

        for i in 0..order {
            let cur = if wb {
                i32::from(SILK_LSF_CODEBOOK_WB[lsf_i1][i])
            } else {
                i32::from(SILK_LSF_CODEBOOK_NBMB[lsf_i1][i])
            };
            let weight = if wb {
                i32::from(SILK_MODEL_LSF_WEIGHT_WB[lsf_i1][i])
            } else {
                i32::from(SILK_MODEL_LSF_WEIGHT_NBMB[lsf_i1][i])
            };
            let value = cur * 128 + (lsf_res[i] * 16384) / weight;
            nlsf[i] = value.clamp(0, 32767) as i16;
        }

        stabilize_lsf(
            &mut nlsf,
            order,
            if wb {
                &SILK_LSF_MIN_SPACING_WB[..]
            } else {
                &SILK_LSF_MIN_SPACING_NBMB[..]
            },
        );

        let frame = &mut self.frame[channel];
        let mut has_lpc_leadin = false;
        if self.subframes == 4 {
            let mut offset = rc.dec_cdf(&SILK_MODEL_LSF_INTERPOLATION_OFFSET) as i32;
            if offset != 4 && frame.coded {
                has_lpc_leadin = true;
                if offset != 0 {
                    let mut nlsf_leadin = [0i16; 16];
                    for i in 0..order {
                        nlsf_leadin[i] = (i32::from(frame.nlsf[i])
                            + (((i32::from(nlsf[i]) - i32::from(frame.nlsf[i])) * offset) >> 2))
                            as i16;
                    }
                    lsf2lpc(&nlsf_leadin, lpc_leadin, order);
                } else {
                    *lpc_leadin = frame.lpc;
                }
            } else {
                offset = 4;
            }
            self.nlsf_interp_factor = offset;
            lsf2lpc(&nlsf, lpc, order);
        } else {
            self.nlsf_interp_factor = 4;
            lsf2lpc(&nlsf, lpc, order);
        }
        frame.nlsf[..order].copy_from_slice(&nlsf[..order]);
        frame.lpc[..order].copy_from_slice(&lpc[..order]);
        (order, has_lpc_leadin)
    }

    /// `silk_decode_excitation`.
    fn decode_excitation(
        &self,
        rc: &mut RangeDecoder,
        excitationf: &mut [f32],
        qoffset_high: usize,
        active: usize,
        voiced: usize,
    ) {
        let mut pulsecount = [0u32; 20];
        let mut lsbcount = [0u32; 20];
        let mut excitation = [0i32; 320];

        let mut seed = rc.dec_cdf(&SILK_MODEL_LCG_SEED);
        let shellblocks = usize::from(SILK_SHELL_BLOCKS[self.bandwidth][self.subframes >> 2]);
        let ratelevel = rc.dec_cdf(&SILK_MODEL_EXC_RATE[voiced]) as usize;

        for i in 0..shellblocks {
            pulsecount[i] = rc.dec_cdf(&SILK_MODEL_PULSE_COUNT[ratelevel]);
            if pulsecount[i] == 17 {
                while pulsecount[i] == 17 {
                    lsbcount[i] += 1;
                    if lsbcount[i] == 10 {
                        break;
                    }
                    pulsecount[i] = rc.dec_cdf(&SILK_MODEL_PULSE_COUNT[9]);
                }
                if lsbcount[i] == 10 {
                    pulsecount[i] = rc.dec_cdf(&SILK_MODEL_PULSE_COUNT[10]);
                }
            }
        }

        for i in 0..shellblocks {
            if pulsecount[i] != 0 {
                let mut branch = [[0i32; 2]; 4];
                let mut loc = 16 * i;
                branch[0][0] = pulsecount[i] as i32;
                let mut b1 = [0i32; 2];
                count_children(rc, 0, branch[0][0], &mut b1);
                branch[1] = b1;
                for b in 0..2 {
                    let mut b2 = [0i32; 2];
                    count_children(rc, 1, branch[1][b], &mut b2);
                    branch[2] = b2;
                    for c in 0..2 {
                        let mut b3 = [0i32; 2];
                        count_children(rc, 2, branch[2][c], &mut b3);
                        branch[3] = b3;
                        for d in 0..2 {
                            count_children(rc, 3, branch[3][d], &mut excitation[loc..loc + 2]);
                            loc += 2;
                        }
                    }
                }
            } else {
                excitation[16 * i..16 * i + 16].fill(0);
            }
        }

        for i in 0..shellblocks << 4 {
            for _ in 0..lsbcount[i >> 4] {
                excitation[i] =
                    (excitation[i] << 1) | rc.dec_cdf(&SILK_MODEL_EXCITATION_LSB) as i32;
            }
        }

        for i in 0..shellblocks << 4 {
            if excitation[i] != 0 {
                let sign = rc.dec_cdf(
                    &SILK_MODEL_EXCITATION_SIGN[active + voiced][qoffset_high]
                        [(pulsecount[i >> 4] as usize).min(6)],
                );
                if sign == 0 {
                    excitation[i] = -excitation[i];
                }
            }
        }

        for i in 0..shellblocks << 4 {
            let value = excitation[i];
            excitation[i] =
                value.wrapping_mul(256) | i32::from(SILK_QUANT_OFFSET[voiced][qoffset_high]);
            if value < 0 {
                excitation[i] += 20;
            } else if value > 0 {
                excitation[i] -= 20;
            }
            seed = seed.wrapping_mul(196_314_165).wrapping_add(907_633_515);
            if seed & 0x8000_0000 != 0 {
                excitation[i] = excitation[i].wrapping_neg();
            }
            seed = seed.wrapping_add(value as u32);
            excitationf[i] = excitation[i] as f32 / 8_388_608.0;
        }
    }

    /// `silk_decode_frame`.
    #[allow(clippy::too_many_arguments)]
    fn decode_frame(
        &mut self,
        rc: &mut RangeDecoder,
        frame_num: usize,
        channel: usize,
        coded_channels: usize,
        active: usize,
        active1: usize,
        redundant: bool,
    ) {
        let mut lpc_leadin = [0f32; 16];
        let mut lpc_body = [0f32; 16];
        let mut residual = [0f32; SILK_MAX_LAG + SILK_HISTORY];
        let mut sf = [Subframe::default(); 4];

        if coded_channels == 2 && channel == 0 {
            let n = rc.dec_cdf(&SILK_MODEL_STEREO_S1) as usize;
            let mut wi = [0usize; 2];
            let mut ws = [0i32; 2];
            wi[0] = rc.dec_cdf(&SILK_MODEL_STEREO_S2) as usize + 3 * (n / 5);
            ws[0] = rc.dec_cdf(&SILK_MODEL_STEREO_S3) as i32;
            wi[1] = rc.dec_cdf(&SILK_MODEL_STEREO_S2) as usize + 3 * (n % 5);
            ws[1] = rc.dec_cdf(&SILK_MODEL_STEREO_S3) as i32;
            let mut w = [0i32; 2];
            for i in 0..2 {
                let a = i32::from(SILK_STEREO_WEIGHTS[wi[i]]);
                let b = i32::from(SILK_STEREO_WEIGHTS[wi[i] + 1]);
                w[i] = a + (((b - a) * 6554) >> 16) * (ws[i] * 2 + 1);
            }
            self.stereo_weights[0] = (f64::from(w[0] - w[1]) / 8192.0) as f32;
            self.stereo_weights[1] = (f64::from(w[1]) / 8192.0) as f32;
            self.midonly = if active1 != 0 {
                false
            } else {
                rc.dec_cdf(&SILK_MODEL_MID_ONLY) != 0
            };
        }

        let (qoffset_high, voiced);
        if active == 0 {
            qoffset_high = rc.dec_cdf(&SILK_MODEL_FRAME_TYPE_INACTIVE) as usize;
            voiced = 0usize;
        } else {
            let t = rc.dec_cdf(&SILK_MODEL_FRAME_TYPE_ACTIVE) as usize;
            qoffset_high = t & 1;
            voiced = t >> 1;
        }

        for i in 0..self.subframes {
            let frame = &mut self.frame[channel];
            let mut log_gain;
            if i == 0 && (frame_num == 0 || !frame.coded) {
                let x = rc.dec_cdf(&SILK_MODEL_GAIN_HIGHBITS[active + voiced]) as i32;
                log_gain = (x << 3) | rc.dec_cdf(&SILK_MODEL_GAIN_LOWBITS) as i32;
                if frame.coded {
                    log_gain = log_gain.max(frame.log_gain - 16);
                }
            } else {
                let delta_gain = rc.dec_cdf(&SILK_MODEL_GAIN_DELTA) as i32;
                log_gain = ((delta_gain << 1) - 16)
                    .max(frame.log_gain + delta_gain - 4)
                    .clamp(0, 63);
            }
            frame.log_gain = log_gain;

            let lg = ((log_gain * 0x1D1C71) >> 16) + 2090;
            let ipart = lg >> 7;
            let fpart = lg & 127;
            let lingain = (1i32 << ipart)
                + (((-174 * fpart * (128 - fpart)) >> 16) + fpart) * ((1i32 << ipart) >> 7);
            sf[i].gain = lingain as f32 / 65536.0;
        }

        let (order, has_lpc_leadin) =
            self.decode_lpc(channel, rc, &mut lpc_leadin, &mut lpc_body, voiced);

        let bw = self.bandwidth;
        if voiced != 0 {
            let frame = &mut self.frame[channel];
            let mut lag_absolute = frame_num == 0 || !frame.prev_voiced;
            let mut primarylag = 0i32;
            if !lag_absolute {
                let delta = rc.dec_cdf(&SILK_MODEL_PITCH_DELTA) as i32;
                if delta != 0 {
                    primarylag = frame.primarylag + delta - 9;
                } else {
                    lag_absolute = true;
                }
            }
            if lag_absolute {
                let highbits = rc.dec_cdf(&SILK_MODEL_PITCH_HIGHBITS) as i32;
                let lowbits = match bw {
                    0 => rc.dec_cdf(&SILK_MODEL_LCG_SEED),
                    1 => rc.dec_cdf(&SILK_MODEL_PITCH_LOWBITS_MB),
                    _ => rc.dec_cdf(&SILK_MODEL_GAIN_LOWBITS),
                } as i32;
                primarylag = i32::from(SILK_PITCH_MIN_LAG[bw])
                    + highbits * i32::from(SILK_PITCH_SCALE[bw])
                    + lowbits;
            }
            frame.primarylag = primarylag;

            let offsets: &[i8] = if self.subframes == 2 {
                if bw == BANDWIDTH_NARROWBAND {
                    &SILK_PITCH_OFFSET_NB10MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_NB10MS) as usize]
                } else {
                    &SILK_PITCH_OFFSET_MBWB10MS
                        [rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_MBWB10MS) as usize]
                }
            } else if bw == BANDWIDTH_NARROWBAND {
                &SILK_PITCH_OFFSET_NB20MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_NB20MS) as usize]
            } else {
                &SILK_PITCH_OFFSET_MBWB20MS[rc.dec_cdf(&SILK_MODEL_PITCH_CONTOUR_MBWB20MS) as usize]
            };
            for i in 0..self.subframes {
                sf[i].pitchlag = (primarylag + i32::from(offsets[i])).clamp(
                    i32::from(SILK_PITCH_MIN_LAG[bw]),
                    i32::from(SILK_PITCH_MAX_LAG[bw]),
                );
            }

            let ltpfilter = rc.dec_cdf(&SILK_MODEL_LTP_FILTER) as usize;
            for s in sf.iter_mut().take(self.subframes) {
                let index = match ltpfilter {
                    0 => rc.dec_cdf(&SILK_MODEL_LTP_FILTER0_SEL),
                    1 => rc.dec_cdf(&SILK_MODEL_LTP_FILTER1_SEL),
                    _ => rc.dec_cdf(&SILK_MODEL_LTP_FILTER2_SEL),
                } as usize;
                let taps: &[i8; 5] = match ltpfilter {
                    0 => &SILK_LTP_FILTER0_TAPS[index],
                    1 => &SILK_LTP_FILTER1_TAPS[index],
                    _ => &SILK_LTP_FILTER2_TAPS[index],
                };
                for j in 0..5 {
                    s.ltptaps[j] = f32::from(taps[j]) / 128.0;
                }
            }
        }

        let ltpscale = if voiced != 0 && frame_num == 0 {
            f32::from(SILK_LTP_SCALE_FACTOR[rc.dec_cdf(&SILK_MODEL_LTP_SCALE_INDEX) as usize])
                / 16384.0
        } else {
            15565.0f32 / 16384.0
        };

        self.decode_excitation(
            rc,
            &mut residual[SILK_MAX_LAG..],
            qoffset_high,
            active,
            voiced,
        );

        if self.output_channels == channel || redundant {
            return;
        }

        let sflength = self.sflength;
        let flength = self.flength;
        let nlsf_interp_factor = self.nlsf_interp_factor;
        let frame = &mut self.frame[channel];
        for i in 0..self.subframes {
            let lpc_coeff = if i < 2 && has_lpc_leadin {
                &lpc_leadin
            } else {
                &lpc_body
            };
            let dst0 = SILK_HISTORY + i * sflength;
            let res0 = SILK_MAX_LAG + i * sflength;
            let lpc0 = SILK_HISTORY + i * sflength;

            if voiced != 0 {
                let (out_end, scale) = if i < 2 || nlsf_interp_factor == 4 {
                    (-((i * sflength) as isize), ltpscale)
                } else {
                    (-(((i - 2) * sflength) as isize), 1.0f32)
                };
                let start = -(sf[i].pitchlag as isize) - (LTP_ORDER / 2) as isize;
                for j in start..out_end {
                    let base = dst0 as isize + j;
                    let sum = ordered_dot(frame.output[base as usize], order, |k| {
                        (
                            -lpc_coeff[k],
                            frame.output[(base - k as isize - 1) as usize],
                        )
                    });
                    residual[(res0 as isize + j) as usize] =
                        clipf(sum, -1.0, 1.0) * scale / sf[i].gain;
                }
                if out_end != 0 {
                    let rescale = sf[i - 1].gain / sf[i].gain;
                    for j in out_end..0 {
                        residual[(res0 as isize + j) as usize] *= rescale;
                    }
                }
                for j in 0..sflength {
                    // Five taps, each a fused multiply-add in FFmpeg's build.
                    let mut sum = residual[res0 + j];
                    for k in 0..LTP_ORDER {
                        let idx = res0 as isize + j as isize - sf[i].pitchlag as isize
                            + (LTP_ORDER / 2) as isize
                            - k as isize;
                        sum = sf[i].ltptaps[k].mul_add(residual[idx as usize], sum);
                    }
                    residual[res0 + j] = sum;
                }
            }

            for j in 0..sflength {
                let sum = ordered_dot(residual[res0 + j] * sf[i].gain, order, |k| {
                    (lpc_coeff[k], frame.lpc_history[lpc0 + j - k - 1])
                });
                frame.lpc_history[lpc0 + j] = sum;
                frame.output[dst0 + j] = clipf(sum, -1.0, 1.0);
            }
        }

        frame.prev_voiced = voiced != 0;
        frame
            .lpc_history
            .copy_within(flength..flength + SILK_HISTORY, 0);
        frame.output.copy_within(flength..flength + SILK_HISTORY, 0);
        frame.coded = true;
    }

    /// `silk_unmix_ms`.
    fn unmix_ms(&mut self, l: &mut [f32], r: &mut [f32]) {
        let base = SILK_HISTORY - self.flength;
        let mid = &self.frame[0].output;
        let side = &self.frame[1].output;
        let w0_prev = self.prev_stereo_weights[0];
        let w1_prev = self.prev_stereo_weights[1];
        let w0 = self.stereo_weights[0];
        let w1 = self.stereo_weights[1];
        let n1 = SILK_STEREO_INTERP_LEN[self.bandwidth] as usize;
        let at = |buf: &[f32; 2 * SILK_HISTORY], i: isize| buf[(base as isize + i) as usize];
        let mut i = 0usize;
        while i < n1 {
            let ii = i as isize;
            let interp0 = w0_prev + i as f32 * (w0 - w0_prev) / n1 as f32;
            let interp1 = w1_prev + i as f32 * (w1 - w1_prev) / n1 as f32;
            let p0 = 0.25 * (at(mid, ii - 2) + 2.0 * at(mid, ii - 1) + at(mid, ii));
            // ((a * b + c) + d * e): two fused multiply-adds in FFmpeg's build.
            l[i] = clipf(
                interp0.mul_add(
                    p0,
                    (1.0 + interp1).mul_add(at(mid, ii - 1), at(side, ii - 1)),
                ),
                -1.0,
                1.0,
            );
            r[i] = clipf(
                (-interp0).mul_add(
                    p0,
                    (1.0 - interp1).mul_add(at(mid, ii - 1), -at(side, ii - 1)),
                ),
                -1.0,
                1.0,
            );
            i += 1;
        }
        while i < self.flength {
            let ii = i as isize;
            let p0 = 0.25 * (at(mid, ii - 2) + 2.0 * at(mid, ii - 1) + at(mid, ii));
            l[i] = clipf(
                w0.mul_add(p0, (1.0 + w1).mul_add(at(mid, ii - 1), at(side, ii - 1))),
                -1.0,
                1.0,
            );
            r[i] = clipf(
                (-w0).mul_add(p0, (1.0 - w1).mul_add(at(mid, ii - 1), -at(side, ii - 1))),
                -1.0,
                1.0,
            );
            i += 1;
        }
        self.prev_stereo_weights = self.stereo_weights;
    }

    /// `ff_silk_decode_superframe`: decodes into `output[c][..]` and returns
    /// the sample count (at the SILK rate), or `None` for invalid
    /// parameters.
    pub(crate) fn decode_superframe(
        &mut self,
        rc: &mut RangeDecoder,
        output: &mut [[f32; 960]; 2],
        bandwidth: usize,
        coded_channels: usize,
        duration_ms: usize,
    ) -> Option<usize> {
        if bandwidth > BANDWIDTH_WIDEBAND || coded_channels > 2 || duration_ms > 60 {
            return None;
        }
        let mut active = [[0usize; 6]; 2];
        let mut redundancy = [0u32; 2];
        let nb_frames = 1 + usize::from(duration_ms > 20) + usize::from(duration_ms > 40);
        self.subframes = duration_ms / nb_frames / 5;
        self.sflength = 20 * (bandwidth + 2);
        self.flength = self.sflength * self.subframes;
        self.bandwidth = bandwidth;
        self.wb = bandwidth == BANDWIDTH_WIDEBAND;

        if coded_channels > self.prev_coded_channels {
            self.frame[1].flush();
        }
        self.prev_coded_channels = coded_channels;

        for i in 0..coded_channels {
            for j in 0..nb_frames {
                active[i][j] = rc.dec_log(1) as usize;
            }
            redundancy[i] = rc.dec_log(1);
        }
        for r in redundancy.iter_mut().take(coded_channels) {
            if *r != 0 && duration_ms > 20 {
                *r = rc.dec_cdf(if duration_ms == 40 {
                    &SILK_MODEL_LBRR_FLAGS_40[..]
                } else {
                    &SILK_MODEL_LBRR_FLAGS_60[..]
                });
            }
        }

        for i in 0..nb_frames {
            for j in 0..coded_channels {
                if redundancy[j] & (1 << i) != 0 {
                    let active1 = if j == 0 && redundancy[1] & (1 << i) == 0 {
                        0
                    } else {
                        1
                    };
                    self.decode_frame(rc, i, j, coded_channels, 1, active1, true);
                }
            }
            self.midonly = false;
        }

        for i in 0..nb_frames {
            let mut j = 0;
            while j < coded_channels && !self.midonly {
                let active1 = if coded_channels > 1 { active[1][i] } else { 0 };
                self.decode_frame(rc, i, j, coded_channels, active[j][i], active1, false);
                j += 1;
            }
            if self.midonly && self.frame[1].coded {
                self.frame[1].flush();
            }
            let fl = self.flength;
            if coded_channels == 1 || self.output_channels == 1 {
                for out in output.iter_mut().take(self.output_channels) {
                    let src = SILK_HISTORY - fl - 2;
                    out[i * fl..(i + 1) * fl].copy_from_slice(&self.frame[0].output[src..src + fl]);
                }
            } else {
                let (o0, o1) = output.split_at_mut(1);
                let (l, r) = (
                    &mut o0[0][i * fl..(i + 1) * fl],
                    &mut o1[0][i * fl..(i + 1) * fl],
                );
                self.unmix_ms(l, r);
            }
            self.midonly = false;
        }
        Some(nb_frames * self.flength)
    }
}
