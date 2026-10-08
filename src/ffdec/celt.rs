// Port of FFmpeg's Opus CELT decoder (FFmpeg commit 2da55bf:
// libavcodec/opus/dec_celt.c, celt.c, celt.h, dsp.c; the float_dsp helpers
// it calls from libavutil/float_dsp.c).
// Copyright (c) 2012 Andrew D'Addesio, (c) 2013-2014 Mozilla Corporation,
// (c) 2016 Rostislav Pehlivanov <atomnuker@gmail.com>, (c) 2005 Balatoni
// Denes, (c) 2006 Loren Merritt (float_dsp.c);
// LGPL-2.1-or-later (see LICENSE-LGPL).

use super::mdct::Imdct;
use super::pvq::{quant_band, Bands, Low, PvqState, CELT_MAX_BANDS, SPREAD_AGGRESSIVE};
use super::rc::{av_log2, RangeDecoder};
use super::tab::*;

const CELT_SHORT_BLOCKSIZE: usize = 120;
const CELT_OVERLAP: usize = CELT_SHORT_BLOCKSIZE;
const CELT_MAX_LOG_BLOCKS: usize = 3;
const CELT_MAX_FRAME_SIZE: usize = CELT_SHORT_BLOCKSIZE << CELT_MAX_LOG_BLOCKS;
const CELT_VECTORS: i32 = 11;
const CELT_ALLOC_STEPS: i32 = 6;
const CELT_FINE_OFFSET: i32 = 21;
const CELT_MAX_FINE_BITS: i32 = 8;
const CELT_POSTFILTER_MINPERIOD: i32 = 15;
const CELT_ENERGY_SILENCE: f32 = -28.0;
const SPREAD_NORMAL: u32 = 2;

/// `ff_celt_window` = `ff_celt_window_padded + 8`.
fn window(i: usize) -> f32 {
    CELT_WINDOW_PADDED[8 + i]
}

#[derive(Clone)]
struct CeltBlock {
    energy: [f32; CELT_MAX_BANDS],
    prev_energy: [[f32; CELT_MAX_BANDS]; 2],
    collapse_masks: [u8; CELT_MAX_BANDS],
    buf: Vec<f32>,
    coeffs: Vec<f32>,
    pf_period_new: i32,
    pf_gains_new: [f32; 3],
    pf_period: i32,
    pf_gains: [f32; 3],
    pf_period_old: i32,
    pf_gains_old: [f32; 3],
    emph_coeff: f32,
}

impl CeltBlock {
    fn new() -> Self {
        Self {
            energy: [0.0; CELT_MAX_BANDS],
            prev_energy: [[0.0; CELT_MAX_BANDS]; 2],
            collapse_masks: [0; CELT_MAX_BANDS],
            buf: vec![0.0; 2048],
            coeffs: vec![0.0; CELT_MAX_FRAME_SIZE],
            pf_period_new: 0,
            pf_gains_new: [0.0; 3],
            pf_period: 0,
            pf_gains: [0.0; 3],
            pf_period_old: 0,
            pf_gains_old: [0.0; 3],
            emph_coeff: 0.0,
        }
    }
}

/// `CeltFrame`, decoder side.
pub(crate) struct Celt {
    imdct: [Imdct; 4],
    block: [CeltBlock; 2],
    pvq: PvqState,
    channels: usize,
    output_channels: usize,
    size: usize,
    start_band: usize,
    end_band: usize,
    coded_bands: usize,
    transient: bool,
    blocks: usize,
    blocksize: usize,
    silence: bool,
    anticollapse_needed: i32,
    anticollapse: bool,
    dual_stereo: bool,
    flushed: bool,
    alloc_trim: i32,
    framebits: i32,
    remaining: i32,
    caps: [i32; CELT_MAX_BANDS],
    fine_bits: [i32; CELT_MAX_BANDS],
    fine_priority: [i32; CELT_MAX_BANDS],
    pulses: [i32; CELT_MAX_BANDS],
}

/// `vector_fmul_window` (float_dsp.c).
fn vector_fmul_window(buf: &mut [f32], len: usize) {
    // dst = src0 = buf[0..2len), src1 = buf[len..], win = ff_celt_window.
    for k in 0..len {
        let s0 = buf[k];
        let s1 = buf[2 * len - 1 - k];
        let wi = window(k);
        let wj = window(2 * len - 1 - k);
        buf[k] = s0 * wj - s1 * wi;
        buf[2 * len - 1 - k] = s0 * wi + s1 * wj;
    }
}

impl Celt {
    /// `ff_celt_init` (with `ff_celt_flush`).
    pub(crate) fn new(output_channels: usize, apply_phase_inv: bool) -> Self {
        let scale = -1.0 / 32768.0;
        let mut c = Self {
            imdct: [
                Imdct::new(120, scale),
                Imdct::new(240, scale),
                Imdct::new(480, scale),
                Imdct::new(960, scale),
            ],
            block: [CeltBlock::new(), CeltBlock::new()],
            pvq: PvqState::new(apply_phase_inv),
            channels: 0,
            output_channels,
            size: 0,
            start_band: 0,
            end_band: 0,
            coded_bands: 0,
            transient: false,
            blocks: 0,
            blocksize: 0,
            silence: false,
            anticollapse_needed: 0,
            anticollapse: false,
            dual_stereo: false,
            flushed: false,
            alloc_trim: 0,
            framebits: 0,
            remaining: 0,
            caps: [0; CELT_MAX_BANDS],
            fine_bits: [0; CELT_MAX_BANDS],
            fine_priority: [0; CELT_MAX_BANDS],
            pulses: [0; CELT_MAX_BANDS],
        };
        c.flush();
        c
    }

    /// `ff_celt_flush`.
    pub(crate) fn flush(&mut self) {
        if self.flushed {
            return;
        }
        for block in &mut self.block {
            block.prev_energy = [[CELT_ENERGY_SILENCE; CELT_MAX_BANDS]; 2];
            block.energy = [0.0; CELT_MAX_BANDS];
            block.buf.fill(0.0);
            block.pf_gains = [0.0; 3];
            block.pf_gains_old = [0.0; 3];
            block.pf_gains_new = [0.0; 3];
            block.emph_coeff = 0.0 / OPUS_DEEMPH_WEIGHT;
        }
        self.pvq.seed = 0;
        self.flushed = true;
    }

    fn decode_coarse_energy(&mut self, rc: &mut RangeDecoder) {
        let mut prev = [0f32; 2];
        let mut alpha = CELT_ALPHA_COEF[self.size];
        let mut beta = CELT_BETA_COEF[self.size];
        let mut model: &[u8] = &CELT_COARSE_ENERGY_DIST[self.size][0];
        if rc.tell() as i32 + 3 <= self.framebits && rc.dec_log(3) != 0 {
            alpha = 0.0;
            beta = 1.0 - (4915.0 / 32768.0);
            model = &CELT_COARSE_ENERGY_DIST[self.size][1];
        }
        for i in 0..CELT_MAX_BANDS {
            for j in 0..self.channels {
                let block = &mut self.block[j];
                if i < self.start_band || i >= self.end_band {
                    block.energy[i] = 0.0;
                    continue;
                }
                let available = self.framebits - rc.tell() as i32;
                let value: f32 = if available >= 15 {
                    let k = i.min(20) << 1;
                    rc.dec_laplace(u32::from(model[k]) << 7, i32::from(model[k + 1]) << 6) as f32
                } else if available >= 2 {
                    let x = rc.dec_cdf(&CELT_MODEL_TAPSET) as i32;
                    ((x >> 1) ^ -(x & 1)) as f32
                } else if available >= 1 {
                    -(rc.dec_log(1) as f32)
                } else {
                    -1.0
                };
                block.energy[i] = block.energy[i].max(-9.0) * alpha + prev[j] + value;
                prev[j] += beta * value;
            }
        }
    }

    fn decode_fine_energy(&mut self, rc: &mut RangeDecoder) {
        for i in self.start_band..self.end_band {
            if self.fine_bits[i] == 0 {
                continue;
            }
            for j in 0..self.channels {
                let q2 = rc.get_raw(self.fine_bits[i] as u32);
                let offset =
                    (q2 as f32 + 0.5) * (1i32 << (14 - self.fine_bits[i])) as f32 / 16384.0 - 0.5;
                self.block[j].energy[i] += offset;
            }
        }
    }

    fn decode_final_energy(&mut self, rc: &mut RangeDecoder) {
        let mut bits_left = self.framebits - rc.tell() as i32;
        for priority in 0..2 {
            let mut i = self.start_band;
            while i < self.end_band && bits_left >= self.channels as i32 {
                if self.fine_priority[i] != priority || self.fine_bits[i] >= CELT_MAX_FINE_BITS {
                    i += 1;
                    continue;
                }
                for j in 0..self.channels {
                    let q2 = rc.get_raw(1);
                    let offset =
                        (q2 as f32 - 0.5) * (1i32 << (14 - self.fine_bits[i] - 1)) as f32 / 16384.0;
                    self.block[j].energy[i] += offset;
                    bits_left -= 1;
                }
                i += 1;
            }
        }
    }

    fn decode_tf_changes(&mut self, rc: &mut RangeDecoder) {
        let mut diff = 0i32;
        let mut tf_select = 0usize;
        let mut tf_changed = 0i32;
        let mut bits: u32 = if self.transient { 2 } else { 4 };
        let mut consumed = rc.tell() as i32;
        let tf_select_bit = self.size != 0 && consumed + (bits as i32) < self.framebits;
        for i in self.start_band..self.end_band {
            if consumed + bits as i32 + i32::from(tf_select_bit) <= self.framebits {
                diff ^= rc.dec_log(bits) as i32;
                consumed = rc.tell() as i32;
                tf_changed |= diff;
            }
            self.pvq.tf_change[i] = diff;
            bits = if self.transient { 4 } else { 5 };
        }
        let t = usize::from(self.transient);
        let tc = tf_changed as usize;
        if tf_select_bit
            && CELT_TF_SELECT[self.size][t][0][tc] != CELT_TF_SELECT[self.size][t][1][tc]
        {
            tf_select = rc.dec_log(1) as usize;
        }
        for i in self.start_band..self.end_band {
            let ch = self.pvq.tf_change[i] as usize;
            self.pvq.tf_change[i] = i32::from(CELT_TF_SELECT[self.size][t][tf_select][ch]);
        }
    }

    fn normc(&self, bits: i32) -> i32 {
        (bits << (self.channels - 1) << self.size) >> 2
    }

    /// `ff_celt_bitalloc` (decoding).
    fn bitalloc(&mut self, rc: &mut RangeDecoder) {
        let ch = self.channels as i32;
        let size = self.size as i32;
        let start = self.start_band;
        let end = self.end_band;
        let mut skip_startband = start;
        let mut skip_bit = 0;
        let mut intensitystereo_bit = 0;
        let mut dualstereo_bit = 0;
        let mut dynalloc = 6;
        let mut extrabits = 0;
        let mut boost = [0i32; CELT_MAX_BANDS];
        let mut trim_offset = [0i32; CELT_MAX_BANDS];
        let mut threshold = [0i32; CELT_MAX_BANDS];
        let mut bits1 = [0i32; CELT_MAX_BANDS];
        let mut bits2 = [0i32; CELT_MAX_BANDS];

        if rc.tell() as i32 + 4 <= self.framebits {
            self.pvq.spread = rc.dec_cdf(&CELT_MODEL_SPREAD);
        } else {
            self.pvq.spread = SPREAD_NORMAL;
        }

        for i in 0..CELT_MAX_BANDS {
            self.caps[i] = self.normc(
                (i32::from(CELT_STATIC_CAPS[self.size][self.channels - 1][i]) + 64)
                    * i32::from(CELT_FREQ_RANGE[i]),
            );
        }

        let mut tbits_8ths = self.framebits << 3;
        for i in start..end {
            let mut quanta = i32::from(CELT_FREQ_RANGE[i]) << (ch - 1) << size;
            let mut b_dynalloc = dynalloc;
            quanta = (quanta << 3).min((6 << 3).max(quanta));
            // uint32 + int against int: an unsigned comparison in C.
            while rc.tell_frac().wrapping_add((b_dynalloc << 3) as u32) < tbits_8ths as u32
                && boost[i] < self.caps[i]
            {
                let is_boost = rc.dec_log(b_dynalloc as u32);
                if is_boost == 0 {
                    break;
                }
                boost[i] += quanta;
                tbits_8ths -= quanta;
                b_dynalloc = 1;
            }
            if boost[i] != 0 {
                dynalloc = (dynalloc - 1).max(2);
            }
        }

        self.alloc_trim = 5;
        if rc.tell_frac().wrapping_add(6 << 3) <= tbits_8ths as u32 {
            self.alloc_trim = rc.dec_cdf(&CELT_MODEL_ALLOC_TRIM) as i32;
        }

        tbits_8ths = (self.framebits << 3) - rc.tell_frac() as i32 - 1;
        self.anticollapse_needed = 0;
        if self.transient && self.size >= 2 && tbits_8ths >= ((size + 2) << 3) {
            self.anticollapse_needed = 1 << 3;
        }
        tbits_8ths -= self.anticollapse_needed;

        if tbits_8ths >= 1 << 3 {
            skip_bit = 1 << 3;
        }
        tbits_8ths -= skip_bit;

        if self.channels == 2 {
            intensitystereo_bit = i32::from(CELT_LOG2_FRAC[end - start]);
            if intensitystereo_bit <= tbits_8ths {
                tbits_8ths -= intensitystereo_bit;
                if tbits_8ths >= 1 << 3 {
                    dualstereo_bit = 1 << 3;
                    tbits_8ths -= 1 << 3;
                }
            } else {
                intensitystereo_bit = 0;
            }
        }

        for i in start..end {
            let trim = self.alloc_trim - 5 - size;
            let band = i32::from(CELT_FREQ_RANGE[i]) * (end as i32 - i as i32 - 1);
            let duration = size + 3;
            let scale = duration + ch - 1;
            threshold[i] = ((3 * i32::from(CELT_FREQ_RANGE[i])) << duration >> 4).max(ch << 3);
            trim_offset[i] = (trim * (band << scale)) >> 6;
            if i32::from(CELT_FREQ_RANGE[i]) << size == 1 {
                trim_offset[i] -= ch << 3;
            }
        }

        let mut low = 1;
        let mut high = CELT_VECTORS - 1;
        while low <= high {
            let center = (low + high) >> 1;
            let mut done = false;
            let mut total = 0;
            for i in (start..end).rev() {
                let mut bandbits = self.normc(
                    i32::from(CELT_FREQ_RANGE[i])
                        * i32::from(CELT_STATIC_ALLOC[center as usize][i]),
                );
                if bandbits != 0 {
                    bandbits = (bandbits + trim_offset[i]).max(0);
                }
                bandbits += boost[i];
                if bandbits >= threshold[i] || done {
                    done = true;
                    total += bandbits.min(self.caps[i]);
                } else if bandbits >= ch << 3 {
                    total += ch << 3;
                }
            }
            if total > tbits_8ths {
                high = center - 1;
            } else {
                low = center + 1;
            }
        }
        high = low;
        low -= 1;

        for i in start..end {
            bits1[i] = self.normc(
                i32::from(CELT_FREQ_RANGE[i]) * i32::from(CELT_STATIC_ALLOC[low as usize][i]),
            );
            bits2[i] = if high >= CELT_VECTORS {
                self.caps[i]
            } else {
                self.normc(
                    i32::from(CELT_FREQ_RANGE[i]) * i32::from(CELT_STATIC_ALLOC[high as usize][i]),
                )
            };
            if bits1[i] != 0 {
                bits1[i] = (bits1[i] + trim_offset[i]).max(0);
            }
            if bits2[i] != 0 {
                bits2[i] = (bits2[i] + trim_offset[i]).max(0);
            }
            if low != 0 {
                bits1[i] += boost[i];
            }
            bits2[i] += boost[i];
            if boost[i] != 0 {
                skip_startband = i;
            }
            bits2[i] = (bits2[i] - bits1[i]).max(0);
        }

        let mut low = 0;
        let mut high = 1 << CELT_ALLOC_STEPS;
        for _ in 0..CELT_ALLOC_STEPS {
            let center = (low + high) >> 1;
            let mut done = false;
            let mut total = 0;
            for j in (start..end).rev() {
                let bandbits = bits1[j] + ((center * bits2[j]) >> CELT_ALLOC_STEPS);
                if bandbits >= threshold[j] || done {
                    done = true;
                    total += bandbits.min(self.caps[j]);
                } else if bandbits >= ch << 3 {
                    total += ch << 3;
                }
            }
            if total > tbits_8ths {
                high = center;
            } else {
                low = center;
            }
        }

        let mut done = false;
        let mut total = 0;
        for i in (start..end).rev() {
            let mut bandbits = bits1[i] + ((low * bits2[i]) >> CELT_ALLOC_STEPS);
            if bandbits >= threshold[i] || done {
                done = true;
            } else {
                bandbits = if bandbits >= ch << 3 { ch << 3 } else { 0 };
            }
            bandbits = bandbits.min(self.caps[i]);
            self.pulses[i] = bandbits;
            total += bandbits;
        }

        self.coded_bands = end;
        loop {
            let j = self.coded_bands - 1;
            if j == skip_startband {
                tbits_8ths += skip_bit;
                break;
            }
            let span = i32::from(CELT_FREQ_BANDS[j + 1]) - i32::from(CELT_FREQ_BANDS[start]);
            let mut remaining = tbits_8ths - total;
            let bandbits = remaining / span;
            remaining -= bandbits * span;
            let mut allocation = self.pulses[j] + bandbits * i32::from(CELT_FREQ_RANGE[j]);
            allocation += (remaining
                - (i32::from(CELT_FREQ_BANDS[j]) - i32::from(CELT_FREQ_BANDS[start])))
            .max(0);
            if allocation >= threshold[j].max((ch + 1) << 3) {
                let do_not_skip = rc.dec_log(1);
                if do_not_skip != 0 {
                    break;
                }
                total += 1 << 3;
                allocation -= 1 << 3;
            }
            total -= self.pulses[j];
            if intensitystereo_bit != 0 {
                total -= intensitystereo_bit;
                intensitystereo_bit = i32::from(CELT_LOG2_FRAC[j - start]);
                total += intensitystereo_bit;
            }
            self.pulses[j] = if allocation >= ch << 3 { ch << 3 } else { 0 };
            total += self.pulses[j];
            self.coded_bands -= 1;
        }

        self.pvq.intensity_stereo = 0;
        self.dual_stereo = false;
        if intensitystereo_bit != 0 {
            self.pvq.intensity_stereo =
                start as i32 + rc.dec_uint((self.coded_bands + 1 - start) as u32) as i32;
        }
        if self.pvq.intensity_stereo <= start as i32 {
            tbits_8ths += dualstereo_bit;
        } else if dualstereo_bit != 0 {
            self.dual_stereo = rc.dec_log(1) != 0;
        }

        let mut remaining = tbits_8ths - total;
        let span = i32::from(CELT_FREQ_BANDS[self.coded_bands]) - i32::from(CELT_FREQ_BANDS[start]);
        let bandbits = remaining / span;
        remaining -= bandbits * span;
        for i in start..self.coded_bands {
            let bits = remaining.min(i32::from(CELT_FREQ_RANGE[i]));
            self.pulses[i] += bits + bandbits * i32::from(CELT_FREQ_RANGE[i]);
            remaining -= bits;
        }

        let mut i = start;
        while i < self.coded_bands {
            let n = i32::from(CELT_FREQ_RANGE[i]) << size;
            let prev_extra = extrabits;
            self.pulses[i] += extrabits;
            if n > 1 {
                extrabits = (self.pulses[i] - self.caps[i]).max(0);
                self.pulses[i] -= extrabits;
                let dof = n * ch
                    + i32::from(
                        ch == 2
                            && n > 2
                            && !self.dual_stereo
                            && (i as i32) < self.pvq.intensity_stereo,
                    );
                let temp = dof * (i32::from(CELT_LOG_FREQ_RANGE[i]) + (size << 3));
                let mut offset = (temp >> 1) - dof * CELT_FINE_OFFSET;
                if n == 2 {
                    offset += dof << 1;
                }
                if self.pulses[i] + offset < 2 * (dof << 3) {
                    offset += temp >> 2;
                } else if self.pulses[i] + offset < 3 * (dof << 3) {
                    offset += temp >> 3;
                }
                let fine_bits = (self.pulses[i] + offset + (dof << 2)) / (dof << 3);
                let max_bits = ((self.pulses[i] >> 3) >> (ch - 1)).clamp(0, CELT_MAX_FINE_BITS);
                self.fine_bits[i] = fine_bits.clamp(0, max_bits);
                self.fine_priority[i] =
                    i32::from(self.fine_bits[i] * (dof << 3) >= self.pulses[i] + offset);
                self.pulses[i] -= self.fine_bits[i] << (ch - 1) << 3;
            } else {
                extrabits = (self.pulses[i] - (ch << 3)).max(0);
                self.pulses[i] -= extrabits;
                self.fine_bits[i] = 0;
                self.fine_priority[i] = 1;
            }
            if extrabits > 0 {
                let mut fineextra =
                    (extrabits >> (ch + 2)).min(CELT_MAX_FINE_BITS - self.fine_bits[i]);
                self.fine_bits[i] += fineextra;
                fineextra <<= ch + 2;
                self.fine_priority[i] = i32::from(fineextra >= extrabits - prev_extra);
                extrabits -= fineextra;
            }
            i += 1;
        }
        self.remaining = extrabits;
        while i < end {
            self.fine_bits[i] = self.pulses[i] >> (ch - 1) >> 3;
            self.pulses[i] = 0;
            self.fine_priority[i] = i32::from(self.fine_bits[i] < 1);
            i += 1;
        }
    }

    /// `ff_celt_quant_bands` (decoding).
    fn quant_bands(&mut self, rc: &mut RangeDecoder) {
        let mut scratch = [0f32; 8 * 22];
        let mut norm = [0f32; 2 * 8 * 100];
        const NORM2: usize = 8 * 100;
        let totalbits = (self.framebits << 3) - self.anticollapse_needed;
        let mut update_lowband = true;
        let mut lowband_offset = 0usize;
        let size = self.size;
        let start = self.start_band;
        let channels = self.channels;

        for i in start..self.end_band {
            let blocks_mask = (1u32 << self.blocks) - 1;
            let mut cm = [blocks_mask, blocks_mask];
            let band_offset = usize::from(CELT_FREQ_BANDS[i]) << size;
            let band_size = usize::from(CELT_FREQ_RANGE[i]) << size;
            let consumed = rc.tell_frac() as i32;
            let mut effective_lowband: isize = -1;
            let mut b = 0;

            if i != start {
                self.remaining -= consumed;
            }
            self.pvq.remaining2 = totalbits - consumed - 1;
            if i < self.coded_bands {
                let curr_balance = self.remaining / (self.coded_bands as i32 - i as i32).min(3);
                b = (self.pvq.remaining2 + 1)
                    .min(self.pulses[i] + curr_balance)
                    .clamp(0, 16383);
            }

            if (i32::from(CELT_FREQ_BANDS[i]) - i32::from(CELT_FREQ_RANGE[i])
                >= i32::from(CELT_FREQ_BANDS[start])
                || i == start + 1)
                && (update_lowband || lowband_offset == 0)
            {
                lowband_offset = i;
            }

            if i == start + 1 {
                let count =
                    (usize::from(CELT_FREQ_RANGE[i]) - usize::from(CELT_FREQ_RANGE[i - 1])) << size;
                norm.copy_within(band_offset - count..band_offset, band_offset);
                if channels == 2 {
                    norm.copy_within(
                        NORM2 + band_offset - count..NORM2 + band_offset,
                        NORM2 + band_offset,
                    );
                }
            }

            if lowband_offset != 0
                && (self.pvq.spread != SPREAD_AGGRESSIVE
                    || self.blocks > 1
                    || self.pvq.tf_change[i] < 0)
            {
                let eff = (i32::from(CELT_FREQ_BANDS[start])).max(
                    i32::from(CELT_FREQ_BANDS[lowband_offset]) - i32::from(CELT_FREQ_RANGE[i]),
                );
                effective_lowband = eff as isize;
                let mut foldstart = lowband_offset;
                loop {
                    foldstart -= 1;
                    if i32::from(CELT_FREQ_BANDS[foldstart]) <= eff || foldstart == 0 {
                        break;
                    }
                }
                let mut foldend = lowband_offset - 1;
                loop {
                    foldend += 1;
                    if !(foldend < i
                        && i32::from(CELT_FREQ_BANDS[foldend])
                            < eff + i32::from(CELT_FREQ_RANGE[i]))
                    {
                        break;
                    }
                }
                cm = [0, 0];
                for j in foldstart..foldend {
                    cm[0] |= u32::from(self.block[0].collapse_masks[j]);
                    cm[1] |= u32::from(self.block[channels - 1].collapse_masks[j]);
                }
            }

            if self.dual_stereo && i as i32 == self.pvq.intensity_stereo {
                self.dual_stereo = false;
                for j in (usize::from(CELT_FREQ_BANDS[start]) << size)..band_offset {
                    norm[j] = (norm[j] + norm[NORM2 + j]) / 2.0;
                }
            }

            let (loc1, loc2) = if effective_lowband != -1 {
                let o = (effective_lowband as usize) << size;
                (Some(Low::Norm(o)), Some(Low::Norm(NORM2 + o)))
            } else {
                (None, None)
            };

            let (b0, b1) = self.block.split_at_mut(1);
            let x = &mut b0[0].coeffs[band_offset..band_offset + band_size];
            let mut bufs = Bands {
                norm: &mut norm,
                scratch: &mut scratch,
            };
            if self.dual_stereo {
                let y = &mut b1[0].coeffs[band_offset..band_offset + band_size];
                cm[0] = quant_band(
                    &mut self.pvq,
                    rc,
                    &mut bufs,
                    i,
                    x,
                    None,
                    band_size as i32,
                    b >> 1,
                    self.blocks as u32,
                    loc1,
                    size as i32,
                    Some(band_offset),
                    0,
                    1.0,
                    Some(0),
                    cm[0],
                );
                cm[1] = quant_band(
                    &mut self.pvq,
                    rc,
                    &mut bufs,
                    i,
                    y,
                    None,
                    band_size as i32,
                    b >> 1,
                    self.blocks as u32,
                    loc2,
                    size as i32,
                    Some(NORM2 + band_offset),
                    0,
                    1.0,
                    Some(0),
                    cm[1],
                );
            } else {
                let y = if channels == 2 {
                    Some(&mut b1[0].coeffs[band_offset..band_offset + band_size])
                } else {
                    None
                };
                cm[0] = quant_band(
                    &mut self.pvq,
                    rc,
                    &mut bufs,
                    i,
                    x,
                    y,
                    band_size as i32,
                    b,
                    self.blocks as u32,
                    loc1,
                    size as i32,
                    Some(band_offset),
                    0,
                    1.0,
                    Some(0),
                    cm[0] | cm[1],
                );
                cm[1] = cm[0];
            }

            self.block[0].collapse_masks[i] = cm[0] as u8;
            self.block[channels - 1].collapse_masks[i] = cm[1] as u8;
            self.remaining += self.pulses[i] + consumed;
            update_lowband = b > (band_size as i32) << 3;
        }
    }

    fn process_anticollapse(&mut self, ch: usize) {
        for i in self.start_band..self.end_band {
            let mut renormalize = false;
            let n = usize::from(CELT_FREQ_RANGE[i]) << self.size;
            let depth = (1 + self.pulses[i]) / n as i32;
            let thresh = ((-1.0 - f64::from(0.125f32 * depth as f32)) as f32).exp2();
            let sqrt_1 = 1.0f32 / (n as f32).sqrt();
            let xoff = usize::from(CELT_FREQ_BANDS[i]) << self.size;
            let mut prev = [
                self.block[ch].prev_energy[0][i],
                self.block[ch].prev_energy[1][i],
            ];
            if self.channels == 1 {
                prev[0] = prev[0].max(self.block[1].prev_energy[0][i]);
                prev[1] = prev[1].max(self.block[1].prev_energy[1][i]);
            }
            let ediff = (self.block[ch].energy[i] - prev[0].min(prev[1])).max(0.0);
            let mut r = (1.0 - ediff).exp2();
            if self.size == 3 {
                r = (f64::from(r) * std::f64::consts::SQRT_2) as f32;
            }
            r = thresh.min(r) * sqrt_1;
            for k in 0..1usize << self.size {
                if self.block[ch].collapse_masks[i] & (1 << k) == 0 {
                    for j in 0..usize::from(CELT_FREQ_RANGE[i]) {
                        let v = if self.pvq.rng() & 0x8000 != 0 { r } else { -r };
                        self.block[ch].coeffs[xoff + (j << self.size) + k] = v;
                    }
                    renormalize = true;
                }
            }
            if renormalize {
                let x = &mut self.block[ch].coeffs[xoff..xoff + n];
                let mut g = 1e-15f32;
                for v in x.iter() {
                    g += v * v;
                }
                let g = 1.0 / g.sqrt();
                for v in x.iter_mut() {
                    *v *= g;
                }
            }
        }
    }

    fn denormalize(&mut self, ch: usize) {
        let block = &mut self.block[ch];
        for i in self.start_band..self.end_band {
            let off = usize::from(CELT_FREQ_BANDS[i]) << self.size;
            let log_norm = block.energy[i] + CELT_MEAN_ENERGY[i];
            let norm = log_norm.min(32.0).exp2();
            for v in &mut block.coeffs[off..off + (usize::from(CELT_FREQ_RANGE[i]) << self.size)] {
                *v *= norm;
            }
        }
    }

    fn postfilter_apply_transition(block: &mut CeltBlock, at: usize) {
        let t0 = block.pf_period_old as isize;
        let t1 = block.pf_period as isize;
        if block.pf_gains[0] == 0.0 && block.pf_gains_old[0] == 0.0 {
            return;
        }
        let (g00, g01, g02) = (
            block.pf_gains_old[0],
            block.pf_gains_old[1],
            block.pf_gains_old[2],
        );
        let (g10, g11, g12) = (block.pf_gains[0], block.pf_gains[1], block.pf_gains[2]);
        let d = &mut block.buf;
        let at = at as isize;
        let g = |d: &Vec<f32>, i: isize| d[(at + i) as usize];
        let mut x1 = g(d, -t1 + 1);
        let mut x2 = g(d, -t1);
        let mut x3 = g(d, -t1 - 1);
        let mut x4 = g(d, -t1 - 2);
        for i in 0..CELT_OVERLAP as isize {
            let wf = CELT_WINDOW2[i as usize];
            let w = f64::from(wf);
            let x0 = g(d, i - t1 + 2);
            // `(1.0 - w) * g * x` is double; `w * g * x` is float.
            let acc = (1.0 - w) * f64::from(g00) * f64::from(g(d, i - t0))
                + (1.0 - w) * f64::from(g01) * f64::from(g(d, i - t0 - 1) + g(d, i - t0 + 1))
                + (1.0 - w) * f64::from(g02) * f64::from(g(d, i - t0 - 2) + g(d, i - t0 + 2))
                + f64::from(wf * g10 * x2)
                + f64::from(wf * g11 * (x1 + x3))
                + f64::from(wf * g12 * (x0 + x4));
            let idx = (at + i) as usize;
            d[idx] = (f64::from(d[idx]) + acc) as f32;
            x4 = x3;
            x3 = x2;
            x2 = x1;
            x1 = x0;
        }
    }

    /// `postfilter_c` (dsp.c).
    fn postfilter_dsp(data: &mut [f32], at: usize, period: usize, gains: [f32; 3], len: usize) {
        let (g0, g1, g2) = (gains[0], gains[1], gains[2]);
        let mut x4 = data[at - period - 2];
        let mut x3 = data[at - period - 1];
        let mut x2 = data[at - period];
        let mut x1 = data[at - period + 1];
        for i in 0..len {
            let x0 = data[at + i - period + 2];
            data[at + i] += g0 * x2 + g1 * (x1 + x3) + g2 * (x0 + x4);
            x4 = x3;
            x3 = x2;
            x2 = x1;
            x1 = x0;
        }
    }

    fn postfilter(&mut self, ch: usize) {
        let len = self.blocksize * self.blocks;
        let filter_len = len as isize - 2 * CELT_OVERLAP as isize;
        let block = &mut self.block[ch];
        Self::postfilter_apply_transition(block, 1024);
        block.pf_period_old = block.pf_period;
        block.pf_gains_old = block.pf_gains;
        block.pf_period = block.pf_period_new;
        block.pf_gains = block.pf_gains_new;
        if len > CELT_OVERLAP {
            Self::postfilter_apply_transition(block, 1024 + CELT_OVERLAP);
            if block.pf_gains[0] > f32::EPSILON && filter_len > 0 {
                let (period, gains) = (block.pf_period as usize, block.pf_gains);
                Self::postfilter_dsp(
                    &mut block.buf,
                    1024 + 2 * CELT_OVERLAP,
                    period,
                    gains,
                    filter_len as usize,
                );
            }
            block.pf_period_old = block.pf_period;
            block.pf_gains_old = block.pf_gains;
        }
        block.buf.copy_within(len..len + 1024 + CELT_OVERLAP / 2, 0);
    }

    fn parse_postfilter(&mut self, rc: &mut RangeDecoder, consumed: i32) -> i32 {
        let mut consumed = consumed;
        self.block[0].pf_gains_new = [0.0; 3];
        self.block[1].pf_gains_new = [0.0; 3];
        if self.start_band == 0 && consumed + 16 <= self.framebits {
            let has_postfilter = rc.dec_log(1);
            if has_postfilter != 0 {
                let octave = rc.dec_uint(6);
                let period = (16i32 << octave) + rc.get_raw(4 + octave) as i32 - 1;
                let gain = 0.09375f32 * (rc.get_raw(3) + 1) as f32;
                let tapset = if rc.tell() as i32 + 2 <= self.framebits {
                    rc.dec_cdf(&CELT_MODEL_TAPSET) as usize
                } else {
                    0
                };
                for block in &mut self.block {
                    block.pf_period_new = period.max(CELT_POSTFILTER_MINPERIOD);
                    block.pf_gains_new[0] = gain * CELT_POSTFILTER_TAPS[tapset][0];
                    block.pf_gains_new[1] = gain * CELT_POSTFILTER_TAPS[tapset][1];
                    block.pf_gains_new[2] = gain * CELT_POSTFILTER_TAPS[tapset][2];
                }
            }
            consumed = rc.tell() as i32;
        }
        consumed
    }

    /// `ff_celt_decode_frame`: decodes into `output[c][..frame_size]` for
    /// each output channel.
    pub(crate) fn decode_frame(
        &mut self,
        rc: &mut RangeDecoder,
        output: &mut [&mut [f32]],
        channels: usize,
        frame_size: usize,
        start_band: usize,
        end_band: usize,
    ) -> Result<(), ()> {
        if channels != 1 && channels != 2 {
            return Err(());
        }
        if start_band > end_band || end_band > CELT_MAX_BANDS {
            return Err(());
        }
        self.silence = false;
        self.transient = false;
        self.anticollapse = false;
        self.flushed = false;
        self.channels = channels;
        self.start_band = start_band;
        self.end_band = end_band;
        self.framebits = (rc.raw_bytes() * 8) as i32;

        let size = av_log2((frame_size / CELT_SHORT_BLOCKSIZE) as u32) as usize;
        if size > CELT_MAX_LOG_BLOCKS || frame_size != CELT_SHORT_BLOCKSIZE << size {
            return Err(());
        }
        self.size = size;
        if self.output_channels == 0 {
            self.output_channels = channels;
        }

        for i in 0..self.channels {
            self.block[i].coeffs.fill(0.0);
            self.block[i].collapse_masks = [0; CELT_MAX_BANDS];
        }

        let mut consumed = rc.tell() as i32;
        if consumed >= self.framebits {
            self.silence = true;
        } else if consumed == 1 {
            self.silence = rc.dec_log(15) != 0;
        }
        if self.silence {
            consumed = self.framebits;
            rc.total_bits = rc
                .total_bits
                .wrapping_add((self.framebits as u32).wrapping_sub(rc.tell()));
        }

        consumed = self.parse_postfilter(rc, consumed);

        if self.size != 0 && consumed + 3 <= self.framebits {
            self.transient = rc.dec_log(3) != 0;
        }
        self.blocks = if self.transient { 1 << self.size } else { 1 };
        self.blocksize = frame_size / self.blocks;
        let imdct_idx = if self.transient { 0 } else { self.size };

        if channels == 1 {
            for i in 0..CELT_MAX_BANDS {
                self.block[0].energy[i] = self.block[0].energy[i].max(self.block[1].energy[i]);
            }
        }

        self.decode_coarse_energy(rc);
        self.decode_tf_changes(rc);
        self.bitalloc(rc);
        self.decode_fine_energy(rc);
        self.quant_bands(rc);

        if self.anticollapse_needed != 0 {
            self.anticollapse = rc.get_raw(1) != 0;
        }
        self.decode_final_energy(rc);

        for i in 0..self.channels {
            if self.anticollapse {
                self.process_anticollapse(i);
            }
            self.denormalize(i);
        }

        let mut downmix = false;
        if self.output_channels < self.channels {
            let (b0, b1) = self.block.split_at_mut(1);
            let n = (frame_size + 15) & !15;
            for (d, s) in b0[0].coeffs[..n].iter_mut().zip(&b1[0].coeffs[..n]) {
                *d += *s * 1.0;
            }
            downmix = true;
        } else if self.output_channels > self.channels {
            let (b0, b1) = self.block.split_at_mut(1);
            b1[0].coeffs[..frame_size].copy_from_slice(&b0[0].coeffs[..frame_size]);
        }

        if self.silence {
            for block in &mut self.block {
                block.energy = [CELT_ENERGY_SILENCE; CELT_MAX_BANDS];
                block.coeffs.fill(0.0);
            }
        }

        for i in 0..self.output_channels {
            for j in 0..self.blocks {
                let dst = 1024 + j * self.blocksize;
                let n = self.blocksize;
                let CeltBlock { buf, coeffs, .. } = &mut self.block[i];
                let at = dst + CELT_OVERLAP / 2;
                self.imdct[imdct_idx].run(&mut buf[at..at + n], &coeffs[j..], self.blocks);
                let block = &mut self.block[i];
                vector_fmul_window(&mut block.buf[dst..dst + CELT_OVERLAP], CELT_OVERLAP / 2);
            }
            if downmix {
                for v in &mut self.block[i].buf[1024..1024 + frame_size] {
                    *v *= 0.5;
                }
            }
            self.postfilter(i);

            let block = &mut self.block[i];
            let c = OPUS_DEEMPH_WEIGHT;
            let mut coeff = block.emph_coeff;
            let src = 1024 - frame_size;
            let out = &mut output[i];
            for k in 0..frame_size {
                coeff = block.buf[src + k] + coeff * c;
                out[k] = coeff;
            }
            block.emph_coeff = if coeff.is_normal() { coeff } else { 0.0 };
        }

        if channels == 1 {
            self.block[1].energy = self.block[0].energy;
        }

        for block in &mut self.block {
            if !self.transient {
                block.prev_energy[1] = block.prev_energy[0];
                block.prev_energy[0] = block.energy;
            } else {
                for j in 0..CELT_MAX_BANDS {
                    block.prev_energy[0][j] = block.prev_energy[0][j].min(block.energy[j]);
                }
            }
            for j in 0..self.start_band {
                block.prev_energy[0][j] = CELT_ENERGY_SILENCE;
                block.energy[j] = 0.0;
            }
            for j in self.end_band..CELT_MAX_BANDS {
                block.prev_energy[0][j] = CELT_ENERGY_SILENCE;
                block.energy[j] = 0.0;
            }
        }

        self.pvq.seed = rc.range;
        Ok(())
    }
}
