// Port of FFmpeg's Opus range decoder (FFmpeg commit 2da55bf:
// libavcodec/opus/rc.c, rc.h), decoder side.
// Copyright (c) 2012 Andrew D'Addesio, (c) 2013-2014 Mozilla Corporation,
// (c) 2017 Rostislav Pehlivanov <atomnuker@gmail.com>;
// LGPL-2.1-or-later (see LICENSE-LGPL).

/// `av_log2`: the index of the highest set bit, 0 for 0.
#[inline]
pub(crate) fn av_log2(v: u32) -> u32 {
    31 - (v | 1).leading_zeros()
}

/// `opus_ilog`: the number of bits needed to write `v`.
#[inline]
pub(crate) fn opus_ilog(v: u32) -> u32 {
    av_log2(v) + u32::from(v != 0)
}

/// `ff_sqrt`: the integer square root, rounded down.
#[inline]
pub(crate) fn ff_sqrt(a: u32) -> u32 {
    let mut r = (a as f64).sqrt() as u32;
    while u64::from(r) * u64::from(r) > u64::from(a) {
        r -= 1;
    }
    while u64::from(r + 1) * u64::from(r + 1) <= u64::from(a) {
        r += 1;
    }
    r
}

const RC_SYM: u32 = 8;
const RC_CEIL: u32 = (1 << RC_SYM) - 1;
const RC_TOP: u32 = 1 << 31;
const RC_BOT: u32 = RC_TOP >> RC_SYM;

/// `OpusRangeCoder`, decoder side: the range-coded symbols read forwards
/// from the start of a frame, the raw bits (`RawBitsContext`) backwards from
/// its end.
pub(crate) struct RangeDecoder<'a> {
    /// The frame and every byte after it in the packet: FFmpeg's bit reader
    /// looks past the frame's end into them (its packet padding is zeros).
    data: &'a [u8],
    /// The frame's size in bytes.
    size: usize,
    /// Bit position of the forward reader (`GetBitContext::index`).
    bit: usize,
    raw_pos: usize,
    raw_bytes: u32,
    cachelen: u32,
    cacheval: u32,
    pub(crate) range: u32,
    pub(crate) value: u32,
    pub(crate) total_bits: u32,
}

impl<'a> RangeDecoder<'a> {
    /// `ff_opus_rc_dec_init` on the first `size` bytes of `data`, a frame
    /// followed by the rest of its packet.
    pub(crate) fn new(data: &'a [u8], size: usize) -> Self {
        let size = size.min(data.len());
        let mut rc = Self {
            data,
            size,
            bit: 0,
            raw_pos: size,
            raw_bytes: 0,
            cachelen: 0,
            cacheval: 0,
            range: 128,
            value: 0,
            total_bits: 9,
        };
        rc.value = 127 - rc.get_bits(7);
        rc.normalize();
        rc
    }

    /// `ff_opus_rc_dec_raw_init`: raw bits are read backwards from byte
    /// `end` of the frame, `bytes` of them at most.
    pub(crate) fn raw_init(&mut self, end: usize, bytes: u32) {
        self.raw_pos = end.min(self.data.len());
        self.raw_bytes = bytes.min(self.raw_pos as u32);
        self.cachelen = 0;
        self.cacheval = 0;
    }

    /// Bytes the raw-bit reader may still read (`rb.bytes`).
    pub(crate) fn raw_bytes(&self) -> u32 {
        self.raw_bytes
    }

    /// `get_bits` of FFmpeg's (safe) bit reader: the bits at the current
    /// index, wherever they are; the index stops 8 bits past the frame.
    fn get_bits(&mut self, n: u32) -> u32 {
        let mut v = 0u32;
        for k in 0..n as usize {
            let at = self.bit + k;
            let byte = self.data.get(at >> 3).copied().unwrap_or(0);
            let b = (byte >> (7 - (at & 7))) & 1;
            v = (v << 1) | u32::from(b);
        }
        self.bit = (self.bit + n as usize).min(self.size * 8 + 8);
        v
    }

    fn normalize(&mut self) {
        while self.range <= RC_BOT {
            self.value =
                ((self.value << RC_SYM) | (self.get_bits(RC_SYM) ^ RC_CEIL)) & (RC_TOP - 1);
            self.range <<= RC_SYM;
            self.total_bits += RC_SYM;
        }
    }

    fn update(&mut self, scale: u32, low: u32, high: u32, total: u32) {
        self.value = self
            .value
            .wrapping_sub(scale.wrapping_mul(total.wrapping_sub(high)));
        self.range = if low != 0 {
            scale.wrapping_mul(high.wrapping_sub(low))
        } else {
            self.range
                .wrapping_sub(scale.wrapping_mul(total.wrapping_sub(high)))
        };
        self.normalize();
    }

    /// `opus_rc_tell`: whole bits consumed.
    pub(crate) fn tell(&self) -> u32 {
        self.total_bits
            .wrapping_sub(av_log2(self.range))
            .wrapping_sub(1)
    }

    /// `opus_rc_tell_frac`: bits consumed, in 1/8 bits.
    pub(crate) fn tell_frac(&self) -> u32 {
        let total_bits = self.total_bits << 3;
        let mut rcbuffer = av_log2(self.range) + 1;
        let mut range = self.range >> (rcbuffer - 16);
        for _ in 0..3 {
            range = range.wrapping_mul(range) >> 15;
            let bit = range >> 16;
            rcbuffer = (rcbuffer << 1) | bit;
            range >>= bit;
        }
        total_bits.wrapping_sub(rcbuffer)
    }

    /// `ff_opus_rc_dec_cdf`: `cdf[0]` is the total, then the cumulative
    /// frequencies.
    pub(crate) fn dec_cdf(&mut self, cdf: &[u16]) -> u32 {
        let total = u32::from(cdf[0]);
        let cdf = &cdf[1..];
        let scale = self.range / total.max(1);
        let symbol = (self.value / scale.max(1)).saturating_add(1);
        let symbol = total - symbol.min(total);
        let mut k = 0usize;
        while k < cdf.len() - 1 && u32::from(cdf[k]) <= symbol {
            k += 1;
        }
        let high = u32::from(cdf[k]);
        let low = if k > 0 { u32::from(cdf[k - 1]) } else { 0 };
        self.update(scale, low, high, total);
        k as u32
    }

    /// `ff_opus_rc_dec_log`.
    pub(crate) fn dec_log(&mut self, bits: u32) -> u32 {
        let scale = self.range >> bits;
        let k = if self.value >= scale {
            self.value -= scale;
            self.range -= scale;
            0
        } else {
            self.range = scale;
            1
        };
        self.normalize();
        k
    }

    /// `ff_opus_rc_get_raw`: `count` (≤ 25) raw bits from the end.
    pub(crate) fn get_raw(&mut self, count: u32) -> u32 {
        while self.raw_bytes != 0 && self.cachelen < count {
            self.raw_pos -= 1;
            self.cacheval |= u32::from(self.data[self.raw_pos]) << self.cachelen;
            self.cachelen += 8;
            self.raw_bytes -= 1;
        }
        let value = if count >= 32 {
            self.cacheval
        } else {
            self.cacheval & ((1u32 << count) - 1)
        };
        self.cacheval = if count >= 32 {
            0
        } else {
            self.cacheval >> count
        };
        self.cachelen = self.cachelen.wrapping_sub(count);
        self.total_bits = self.total_bits.wrapping_add(count);
        value
    }

    /// `ff_opus_rc_dec_uint`: uniform over `0..size`.
    pub(crate) fn dec_uint(&mut self, size: u32) -> u32 {
        let bits = opus_ilog(size.wrapping_sub(1));
        let total = if bits > 8 {
            (size.wrapping_sub(1) >> (bits - 8)) + 1
        } else {
            size.max(1)
        };
        let scale = self.range / total;
        let k = (self.value / scale.max(1)).saturating_add(1);
        let k = total - k.min(total);
        self.update(scale, k, k + 1, total);
        if bits > 8 {
            let k = (k << (bits - 8)) | self.get_raw(bits - 8);
            k.min(size.wrapping_sub(1))
        } else {
            k
        }
    }

    /// `ff_opus_rc_dec_uint_step`.
    pub(crate) fn dec_uint_step(&mut self, k0: u32) -> u32 {
        let total = (k0 + 1) * 3 + k0;
        let scale = self.range / total;
        let symbol = (self.value / scale.max(1)).saturating_add(1);
        let symbol = total - symbol.min(total);
        let k = if symbol < (k0 + 1) * 3 {
            symbol / 3
        } else {
            symbol - (k0 + 1) * 2
        };
        let (low, high) = if k <= k0 {
            (3 * k, 3 * (k + 1))
        } else {
            ((k - 1 - k0) + 3 * (k0 + 1), (k - k0) + 3 * (k0 + 1))
        };
        self.update(scale, low, high, total);
        k
    }

    /// `ff_opus_rc_dec_uint_tri`.
    pub(crate) fn dec_uint_tri(&mut self, qn: u32) -> u32 {
        let total = ((qn >> 1) + 1) * ((qn >> 1) + 1);
        let scale = self.range / total;
        let center = (self.value / scale.max(1)).saturating_add(1);
        let center = total - center.min(total);
        let (k, low, symbol);
        if center < total >> 1 {
            k = (ff_sqrt(8 * center + 1) - 1) >> 1;
            low = (k * (k + 1)) >> 1;
            symbol = k + 1;
        } else {
            k = (2 * (qn + 1) - ff_sqrt(8 * (total - center - 1) + 1)) >> 1;
            low = total - (((qn + 1 - k) * (qn + 2 - k)) >> 1);
            symbol = qn + 1 - k;
        }
        self.update(scale, low, low + symbol, total);
        k
    }

    /// `ff_opus_rc_dec_laplace`.
    pub(crate) fn dec_laplace(&mut self, symbol: u32, decay: i32) -> i32 {
        let mut symbol = symbol;
        let mut value: i32 = 0;
        let mut low: u32 = 0;
        let scale = self.range >> 15;
        let center = (self.value / scale.max(1)).saturating_add(1);
        let center = (1u32 << 15) - center.min(1 << 15);
        if center >= symbol {
            value += 1;
            low = symbol;
            symbol = 1 + (((32768 - 32 - symbol) as i64 * (16384 - decay) as i64) >> 15) as u32;
            while symbol > 1 && center >= low + 2 * symbol {
                value += 1;
                symbol *= 2;
                low += symbol;
                symbol = ((((symbol - 2) as i64) * decay as i64) >> 15) as u32 + 1;
            }
            if symbol <= 1 {
                let distance = (center - low) >> 1;
                value += distance as i32;
                low += 2 * distance;
            }
            if center < low + symbol {
                value = -value;
            } else {
                low += symbol;
            }
        }
        self.update(scale, low, (low + symbol).min(32768), 32768);
        value
    }
}
