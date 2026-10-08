// Port of the part of FFmpeg's libswresample the Opus decoder uses (FFmpeg
// commit 2da55bf: libswresample/swresample.c, resample.c,
// resample_template.c; av_bessel_i0 from libavutil/mathematics.c): planar
// float in and out, same channel layout, any input rate to 48 kHz, with the
// decoder's options (filter_size 16; the defaults phase_shift 10,
// linear_interp 1, exact_rational 1, cutoff 0.97, Kaiser window, beta 9),
// as built for arm64 (no x86 `padless` input hold-back).
// Copyright (c) 2004-2012 Michael Niedermayer <michaelni@gmx.at>, bessel
// function (c) 2006 Xiaogang Zhang, (c) 2005-2012 Michael Niedermayer
// (mathematics.c); LGPL-2.1-or-later (see LICENSE-LGPL).

use std::f64::consts::PI;

const FILTER_SIZE: i64 = 16;
const PHASE_SHIFT: i64 = 10;
const KAISER_BETA: f64 = 9.0;

fn eval_poly(coeff: &[f64], x: f64) -> f64 {
    let mut sum = coeff[coeff.len() - 1];
    for &c in coeff[..coeff.len() - 1].iter().rev() {
        sum *= x;
        sum += c;
    }
    sum
}

/// `av_bessel_i0`.
fn bessel_i0(x: f64) -> f64 {
    const P1: [f64; 15] = [
        -2.2335582639474375249e+15,
        -5.5050369673018427753e+14,
        -3.2940087627407749166e+13,
        -8.4925101247114157499e+11,
        -1.1912746104985237192e+10,
        -1.0313066708737980747e+08,
        -5.9545626019847898221e+05,
        -2.4125195876041896775e+03,
        -7.0935347449210549190e+00,
        -1.5453977791786851041e-02,
        -2.5172644670688975051e-05,
        -3.0517226450451067446e-08,
        -2.6843448573468483278e-11,
        -1.5982226675653184646e-14,
        -5.2487866627945699800e-18,
    ];
    const Q1: [f64; 6] = [
        -2.2335582639474375245e+15,
        7.8858692566751002988e+12,
        -1.2207067397808979846e+10,
        1.0377081058062166144e+07,
        -4.8527560179962773045e+03,
        1.0,
    ];
    const P2: [f64; 7] = [
        -2.2210262233306573296e-04,
        1.3067392038106924055e-02,
        -4.4700805721174453923e-01,
        5.5674518371240761397e+00,
        -2.3517945679239481621e+01,
        3.1611322818701131207e+01,
        -9.6090021968656180000e+00,
    ];
    const Q2: [f64; 8] = [
        -5.5194330231005480228e-04,
        3.2547697594819615062e-02,
        -1.1151759188741312645e+00,
        1.3982595353892851542e+01,
        -6.0228002066743340583e+01,
        8.5539563258012929600e+01,
        -3.1446690275135491500e+01,
        1.0,
    ];
    if x == 0.0 {
        return 1.0;
    }
    let x = x.abs();
    if x <= 15.0 {
        let y = x * x;
        eval_poly(&P1, y) / eval_poly(&Q1, y)
    } else {
        let y = 1.0 / x - 1.0 / 15.0;
        let r = eval_poly(&P2, y) / eval_poly(&Q2, y);
        let factor = x.exp() / x.sqrt();
        factor * r
    }
}

fn gcd(a: i64, b: i64) -> i64 {
    if b == 0 {
        a
    } else {
        gcd(b, a % b)
    }
}

/// One output sample: `Σ src(i) * filter[i]`, accumulated as FFmpeg's arm64
/// build does it (libswresample/aarch64 `ff_resample_common_float_neon`):
/// the `filter_length & ~7` (else `& ~3`) leading taps through the NEON
/// kernel, four lanes of fused multiply-adds (lane `l` takes taps `l`,
/// `l + 4`, ...) summed pairwise at the end, then the rest one fused
/// multiply-add at a time.
fn apply_filter(filter: &[f32], src: impl Fn(usize) -> f32) -> f32 {
    let len = filter.len();
    let x8 = len & !7;
    let x4 = len & !3;
    let vector = if x8 >= 8 {
        x8
    } else if x4 >= 4 {
        x4
    } else {
        0
    };
    let mut val = 0f32;
    if vector != 0 {
        let mut acc = [0f32; 4];
        for block in (0..vector).step_by(4) {
            for (l, a) in acc.iter_mut().enumerate() {
                *a = src(block + l).mul_add(filter[block + l], *a);
            }
        }
        val = (acc[0] + acc[1]) + (acc[2] + acc[3]);
    }
    for (i, &f) in filter.iter().enumerate().skip(vector) {
        val = src(i).mul_add(f, val);
    }
    val
}

/// A sample position in the caller's input planes, which `resample` may
/// move backwards over samples it copied into its own buffer.
struct Input<'a> {
    planes: &'a [&'a [f32]],
    pos: isize,
}

impl Input<'_> {
    fn at(&self, ch: usize, i: isize) -> f32 {
        self.planes
            .get(ch)
            .and_then(|p| p.get((self.pos + i) as usize))
            .copied()
            .unwrap_or(0.0)
    }
}

/// `SwrContext` + `ResampleContext` for the Opus decoder's use.
pub(crate) struct Swr {
    channels: usize,
    initialized: bool,
    in_rate: i64,
    // ResampleContext
    phase_count: i64,
    filter_length: i64,
    filter_alloc: i64,
    filter_bank: Vec<f32>,
    src_incr: i64,
    dst_incr: i64,
    dst_incr_div: i64,
    dst_incr_mod: i64,
    index: i64,
    frac: i64,
    // SwrContext
    in_buffer: Vec<Vec<f32>>,
    in_buffer_index: i64,
    in_buffer_count: i64,
    resample_in_constraint: bool,
    flushed: bool,
}

impl Swr {
    /// `swr_alloc` + the options `opus_decode_init` sets: `channels` (1 or
    /// 2) of planar float to 48 kHz.
    pub(crate) fn new(channels: usize) -> Self {
        Self {
            channels,
            initialized: false,
            in_rate: 0,
            phase_count: 0,
            filter_length: 0,
            filter_alloc: 0,
            filter_bank: Vec::new(),
            src_incr: 0,
            dst_incr: 0,
            dst_incr_div: 0,
            dst_incr_mod: 0,
            index: 0,
            frac: 0,
            in_buffer: vec![Vec::new(); channels],
            in_buffer_index: 0,
            in_buffer_count: 0,
            resample_in_constraint: false,
            flushed: false,
        }
    }

    /// `swr_is_initialized`.
    pub(crate) fn is_initialized(&self) -> bool {
        self.initialized
    }

    /// The `in_sample_rate` option.
    pub(crate) fn in_rate(&self) -> i64 {
        self.in_rate
    }

    /// `swr_close`.
    pub(crate) fn close(&mut self) {
        self.initialized = false;
        for b in &mut self.in_buffer {
            b.clear();
        }
        self.in_buffer_index = 0;
        self.in_buffer_count = 0;
        self.resample_in_constraint = false;
        self.flushed = false;
    }

    /// `swr_init` with `in_sample_rate` = `in_rate` (8, 12 or 16 kHz).
    pub(crate) fn init(&mut self, in_rate: i64) {
        self.close();
        self.in_rate = in_rate;
        let out_rate = 48_000i64;
        // resample_init
        let cutoff = 0.97f64;
        let factor = (out_rate as f64 * cutoff / in_rate as f64).min(1.0);
        let mut phase_count = 1i64 << PHASE_SHIFT;
        let mut filter_length = ((FILTER_SIZE as f64 / factor).ceil() as i64).max(1);
        if filter_length > 1 {
            filter_length = (filter_length + 1) & !1;
        }
        let g = gcd(out_rate, in_rate);
        let exact = out_rate / g;
        if exact <= phase_count {
            phase_count = exact;
        }
        let filter_alloc = (filter_length + 7) & !7;
        if self.phase_count != phase_count
            || self.filter_length != filter_length
            || self.filter_bank.is_empty()
        {
            self.phase_count = phase_count;
            self.filter_length = filter_length;
            self.filter_alloc = filter_alloc;
            self.filter_bank = vec![0.0; (filter_alloc * (phase_count + 1)) as usize];
            build_filter(
                &mut self.filter_bank,
                factor,
                filter_length,
                filter_alloc,
                phase_count,
            );
            let fa = filter_alloc as usize;
            let pc = phase_count as usize;
            let (head, tail) = self.filter_bank.split_at_mut(fa * pc + 1);
            tail[..fa - 1].copy_from_slice(&head[..fa - 1]);
            self.filter_bank[fa * pc] = self.filter_bank[fa - 1];
        }
        // av_reduce(out_rate, in_rate * phase_count)
        let (num, den) = (out_rate, in_rate * phase_count);
        let g = gcd(num, den);
        self.src_incr = num / g;
        self.dst_incr = den / g;
        while self.dst_incr < (1 << 20) && self.src_incr < (1 << 20) {
            self.dst_incr *= 2;
            self.src_incr *= 2;
        }
        self.dst_incr_div = self.dst_incr / self.src_incr;
        self.dst_incr_mod = self.dst_incr % self.src_incr;
        self.index = -phase_count * ((filter_length - 1) / 2);
        self.frac = 0;
        self.initialized = true;
    }

    fn ensure_in_buffer(&mut self, size: i64) {
        let size = size.max(0) as usize;
        for b in &mut self.in_buffer {
            if b.len() < size {
                b.resize(size, 0.0);
            }
        }
    }

    /// `resample_flush`: mirror the buffered tail.
    fn flush_input(&mut self) {
        let reflection = (self.in_buffer_count.min(self.filter_length) + 1) / 2;
        self.ensure_in_buffer(self.in_buffer_index + self.in_buffer_count + reflection);
        let base = (self.in_buffer_index + self.in_buffer_count) as usize;
        for ch in &mut self.in_buffer {
            for j in 0..reflection as usize {
                ch[base + j] = ch[base - j - 1];
            }
        }
        self.in_buffer_count += reflection;
    }

    /// `invert_initial_buffer`. Returns `None` for "wait for more data"
    /// (`INT_MAX`), else the samples taken from `input`.
    fn invert_initial_buffer(&mut self, input: &Input, in_count: i64) -> Option<i64> {
        let fl = self.filter_length;
        let num = (in_count + self.in_buffer_count).min(fl + 1);
        if self.index >= 0 {
            return Some(0);
        }
        self.ensure_in_buffer(fl * 2 + 1);
        for n in self.in_buffer_count..num {
            for ch in 0..self.channels {
                let v = input.at(ch, (n - self.in_buffer_count) as isize);
                self.in_buffer[ch][(fl + n) as usize] = v;
            }
        }
        if num < fl + 1 {
            self.in_buffer_count = num;
            self.in_buffer_index = fl;
            return None;
        }
        for n in 1..=fl {
            for ch in &mut self.in_buffer {
                ch[(fl - n) as usize] = ch[(fl + n) as usize];
            }
        }
        let res = num - self.in_buffer_count;
        self.in_buffer_index = fl;
        while self.index < 0 {
            self.in_buffer_index -= 1;
            self.index += self.phase_count;
        }
        self.in_buffer_count = (self.in_buffer_count + fl).max(1 + fl * 2) - self.in_buffer_index;
        Some(res.max(0))
    }

    /// `multiple_resample` (the `resample_common` path; for the Opus rates
    /// `frac` and `dst_incr_mod` stay 0). `src` gives sample `i` of channel
    /// `ch`; returns the samples written to `out[ch][out_pos..]` and the
    /// input consumed.
    fn multiple_resample(
        &mut self,
        out: &mut [&mut [f32]],
        out_pos: usize,
        dst_size: i64,
        src: &dyn Fn(usize, i64) -> f32,
        src_size: i64,
    ) -> (i64, i64) {
        let end_index = (1 + src_size - self.filter_length) * self.phase_count;
        let delta_frac = (end_index - self.index) * self.src_incr - self.frac;
        let delta_n = (delta_frac + self.dst_incr - 1) / self.dst_incr;
        let dst_size = dst_size.min(delta_n).max(0);
        let mut consumed = 0;
        if dst_size > 0 {
            for ch in 0..self.channels {
                let mut index = self.index;
                let mut frac = self.frac;
                let mut sample_index = 0i64;
                while index >= self.phase_count {
                    sample_index += 1;
                    index -= self.phase_count;
                }
                for dst_index in 0..dst_size {
                    let f = (self.filter_alloc * index) as usize;
                    let filter = &self.filter_bank[f..f + self.filter_length as usize];
                    let val = apply_filter(filter, |i| src(ch, sample_index + i as i64));
                    if let Some(o) = out
                        .get_mut(ch)
                        .and_then(|o| o.get_mut(out_pos + dst_index as usize))
                    {
                        *o = val;
                    }
                    frac += self.dst_incr_mod;
                    index += self.dst_incr_div;
                    if frac >= self.src_incr {
                        frac -= self.src_incr;
                        index += 1;
                    }
                    while index >= self.phase_count {
                        sample_index += 1;
                        index -= self.phase_count;
                    }
                }
                if ch + 1 == self.channels {
                    self.frac = frac;
                    self.index = index;
                    consumed = sample_index;
                }
            }
        }
        (dst_size, consumed)
    }

    /// `swr_convert`: resample `in_count` samples of `input` (or flush with
    /// `None`) into at most `out_count` samples of `out`, returning the
    /// count written.
    pub(crate) fn convert(
        &mut self,
        out: &mut [&mut [f32]],
        out_count: usize,
        input: Option<&[&[f32]]>,
        in_count: usize,
    ) -> usize {
        if !self.initialized {
            return 0;
        }
        let empty: [&[f32]; 0] = [];
        let planes: &[&[f32]] = match input {
            Some(p) => p,
            None => {
                if !self.flushed {
                    self.flush_input();
                }
                self.resample_in_constraint = false;
                self.flushed = true;
                &empty
            }
        };
        let in_count = if input.is_some() { in_count as i64 } else { 0 };
        self.resample(out, out_count as i64, Input { planes, pos: 0 }, in_count) as usize
    }

    /// `resample` (swresample.c), with `padless` 0.
    fn resample(
        &mut self,
        out: &mut [&mut [f32]],
        mut out_count: i64,
        mut input: Input,
        mut in_count: i64,
    ) -> i64 {
        let mut ret_sum = 0i64;
        let mut out_pos = 0usize;
        let mut border = match self.invert_initial_buffer(&input, in_count) {
            None => return 0,
            Some(b) => b,
        };
        if border != 0 {
            input.pos += border as isize;
            in_count -= border;
            self.resample_in_constraint = false;
        }
        loop {
            if !self.resample_in_constraint && self.in_buffer_count != 0 {
                let base = self.in_buffer_index;
                let buffers = std::mem::take(&mut self.in_buffer);
                let src = |ch: usize, i: i64| {
                    buffers[ch].get((base + i) as usize).copied().unwrap_or(0.0)
                };
                let (ret, consumed) =
                    self.multiple_resample(out, out_pos, out_count, &src, self.in_buffer_count);
                self.in_buffer = buffers;
                out_count -= ret;
                ret_sum += ret;
                out_pos += ret as usize;
                self.in_buffer_count -= consumed;
                self.in_buffer_index += consumed;
                if in_count == 0 {
                    break;
                }
                if self.in_buffer_count <= border {
                    input.pos -= self.in_buffer_count as isize;
                    in_count += self.in_buffer_count;
                    self.in_buffer_count = 0;
                    self.in_buffer_index = 0;
                    border = 0;
                }
            }

            if (self.flushed || in_count > 0) && self.in_buffer_count == 0 {
                self.in_buffer_index = 0;
                let pos = input.pos;
                let planes = input.planes;
                let src = |ch: usize, i: i64| {
                    planes
                        .get(ch)
                        .and_then(|p| p.get((pos + i as isize) as usize))
                        .copied()
                        .unwrap_or(0.0)
                };
                let (ret, consumed) =
                    self.multiple_resample(out, out_pos, out_count, &src, in_count.max(0));
                out_count -= ret;
                ret_sum += ret;
                out_pos += ret as usize;
                in_count -= consumed;
                input.pos += consumed as isize;
            }

            let size = self.in_buffer_index + self.in_buffer_count + in_count;
            let capacity = self.in_buffer.first().map_or(0, Vec::len) as i64;
            if size > capacity && self.in_buffer_count + in_count <= self.in_buffer_index {
                let (from, count) = (self.in_buffer_index as usize, self.in_buffer_count as usize);
                for ch in &mut self.in_buffer {
                    ch.copy_within(from..from + count, 0);
                }
                self.in_buffer_index = 0;
            } else {
                self.ensure_in_buffer(size);
            }

            if in_count != 0 {
                let mut count = in_count;
                if self.in_buffer_count != 0 && self.in_buffer_count + 2 < count && out_count != 0 {
                    count = self.in_buffer_count + 2;
                }
                let dst = (self.in_buffer_index + self.in_buffer_count) as usize;
                for ch in 0..self.channels {
                    for k in 0..count as usize {
                        let v = input.at(ch, k as isize);
                        self.in_buffer[ch][dst + k] = v;
                    }
                }
                self.in_buffer_count += count;
                in_count -= count;
                border += count;
                input.pos += count as isize;
                self.resample_in_constraint = false;
                if self.in_buffer_count != count || in_count != 0 {
                    continue;
                }
            }
            break;
        }
        self.resample_in_constraint = out_count != 0;
        ret_sum
    }
}

/// `build_filter` for planar float.
fn build_filter(filter: &mut [f32], factor: f64, tap_count: i64, alloc: i64, phase_count: i64) {
    let ph_nb = if phase_count % 2 != 0 {
        phase_count
    } else {
        phase_count / 2 + 1
    };
    let center = (tap_count - 1) / 2;
    let factor = factor.min(1.0);
    let mut tab = vec![0f64; tap_count as usize + 1];
    let mut sin_lut = vec![0f64; ph_nb as usize];
    let mut norm = 0f64;
    if factor == 1.0 {
        for ph in 0..ph_nb {
            sin_lut[ph as usize] = (PI * ph as f64 / phase_count as f64).sin()
                * if center & 1 != 0 { 1.0 } else { -1.0 };
        }
    }
    for ph in 0..ph_nb {
        let mut s = sin_lut[ph as usize];
        for i in 0..tap_count {
            let x = PI * ((i - center) as f64 - ph as f64 / phase_count as f64) * factor;
            let mut y = if x == 0.0 {
                1.0
            } else if factor == 1.0 {
                s / x
            } else {
                x.sin() / x
            };
            let w = 2.0 * x / (factor * tap_count as f64 * PI);
            y *= bessel_i0(KAISER_BETA * (1.0 - w * w).max(0.0).sqrt());
            tab[i as usize] = y;
            s = -s;
            if ph == 0 {
                norm += y;
            }
        }
        for i in 0..tap_count {
            filter[(ph * alloc + i) as usize] = (tab[i as usize] * 1.0 / norm) as f32;
        }
        if phase_count % 2 != 0 {
            continue;
        }
        for i in 0..tap_count {
            filter[((phase_count - ph) * alloc + tap_count - 1 - i) as usize] =
                filter[(ph * alloc + i) as usize];
        }
    }
}
