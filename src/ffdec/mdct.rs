// The inverse MDCT FFmpeg's CELT decoder runs (FFmpeg commit 2da55bf:
// `av_tx_init(AV_TX_FLOAT_MDCT, inv = 1, len, scale)` from libavutil/tx.c,
// tx_template.c): `len` coefficients in, `len` samples out, the middle half
// of the full inverse transform, as `ff_tx_mdct_naive_inv` defines it:
//   dst[i]         =  scale * sum_j src[j] cos(pi (2j+1)(2len - 2i - 1) / (4 len))
//   dst[i + len/2] = -scale * sum_j src[j] cos(pi (2j+1)(3len + 2i + 1) / (4 len))
// for i < len/2. Both halves are the type-IV DCT C of the input read
// backwards, dst[n] = scale * C[len - 1 - n], computed here through a
// complex FFT of len/2 points in double precision.
// Copyright (c) Lynne (tx_template.c); LGPL-2.1-or-later (see LICENSE-LGPL).

use std::f64::consts::PI;

#[derive(Clone, Copy, Default)]
struct C64 {
    re: f64,
    im: f64,
}

impl C64 {
    #[inline]
    fn mul(self, o: C64) -> C64 {
        C64 {
            re: self.re * o.re - self.im * o.im,
            im: self.re * o.im + self.im * o.re,
        }
    }
    #[inline]
    fn add(self, o: C64) -> C64 {
        C64 {
            re: self.re + o.re,
            im: self.im + o.im,
        }
    }
}

/// A mixed-radix (2, 3, 4, 5) forward complex FFT,
/// `X[p] = sum_k x[k] e^{-2 pi i pk / n}`.
struct Fft {
    n: usize,
    factors: Vec<usize>,
    twiddles: Vec<C64>,
}

impl Fft {
    fn new(n: usize) -> Self {
        let mut factors = Vec::new();
        let mut m = n;
        for p in [4usize, 2, 3, 5] {
            while m % p == 0 && m > 1 {
                factors.push(p);
                m /= p;
            }
        }
        assert_eq!(m, 1, "FFT size {n} has a factor other than 2, 3, 5");
        let twiddles = (0..n)
            .map(|k| {
                let a = -2.0 * PI * k as f64 / n as f64;
                C64 {
                    re: a.cos(),
                    im: a.sin(),
                }
            })
            .collect();
        Self {
            n,
            factors,
            twiddles,
        }
    }

    fn run(&self, input: &[C64], out: &mut [C64]) {
        self.rec(out, input, 0, 1, self.n, 0, 1);
    }

    #[allow(clippy::too_many_arguments)]
    fn rec(
        &self,
        out: &mut [C64],
        input: &[C64],
        in_off: usize,
        in_stride: usize,
        n: usize,
        level: usize,
        tw_stride: usize,
    ) {
        if n == 1 {
            out[0] = input[in_off];
            return;
        }
        let p = self.factors[level];
        let m = n / p;
        for q in 0..p {
            self.rec(
                &mut out[q * m..(q + 1) * m],
                input,
                in_off + q * in_stride,
                in_stride * p,
                m,
                level + 1,
                tw_stride * p,
            );
        }
        let mut tmp = [C64::default(); 5];
        for k in 0..m {
            for (q, t) in tmp.iter_mut().enumerate().take(p) {
                // W_n^{qk}
                *t = out[q * m + k].mul(self.twiddles[(q * k * tw_stride) % self.n]);
            }
            for s in 0..p {
                let mut acc = C64::default();
                for (q, t) in tmp.iter().enumerate().take(p) {
                    // W_p^{qs} = W_n^{qs m}
                    acc = acc.add(t.mul(self.twiddles[(q * s * m * tw_stride) % self.n]));
                }
                out[k + s * m] = acc;
            }
        }
    }
}

/// `av_tx` inverse float MDCT of `len` coefficients.
pub(crate) struct Imdct {
    len: usize,
    scale: f64,
    fft: Fft,
    pre: Vec<C64>,
    post: Vec<C64>,
}

impl Imdct {
    pub(crate) fn new(len: usize, scale: f64) -> Self {
        let m = len / 2;
        let n = len as f64;
        let pre = (0..m)
            .map(|k| {
                let a = -PI * k as f64 / n;
                C64 {
                    re: a.cos(),
                    im: a.sin(),
                }
            })
            .collect();
        let post = (0..m)
            .map(|p| {
                let a = -PI * (4 * p + 1) as f64 / (4.0 * n);
                C64 {
                    re: a.cos(),
                    im: a.sin(),
                }
            })
            .collect();
        Self {
            len,
            scale,
            fft: Fft::new(m),
            pre,
            post,
        }
    }

    /// `dst[0..len] = imdct(src[0], src[stride], ...)`.
    pub(crate) fn run(&self, dst: &mut [f32], src: &[f32], stride: usize) {
        let n = self.len;
        let m = n / 2;
        let x = |j: usize| f64::from(src[j * stride]);
        let mut v = vec![C64::default(); m];
        for (k, vk) in v.iter_mut().enumerate() {
            let t = C64 {
                re: x(2 * k),
                im: x(n - 1 - 2 * k),
            };
            *vk = t.mul(self.pre[k]);
        }
        let mut big = vec![C64::default(); m];
        self.fft.run(&v, &mut big);
        // C[2p] = Re U[p], C[n-1-2p] = -Im U[p]; dst[i] = scale * C[n-1-i].
        for p in 0..m {
            let u = big[p].mul(self.post[p]);
            dst[n - 1 - 2 * p] = (self.scale * u.re) as f32;
            dst[2 * p] = (self.scale * -u.im) as f32;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The fast transform equals `ff_tx_mdct_naive_inv`'s definition.
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
            let mut worst = 0f64;
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
                worst = worst.max((f64::from(fast[i]) - d * scale).abs());
                worst = worst.max((f64::from(fast[i + half]) + u * scale).abs());
            }
            assert!(worst < 1e-4, "len {len}: max error {worst}");
        }
    }
}
