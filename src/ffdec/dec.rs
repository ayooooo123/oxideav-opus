// Port of FFmpeg's Opus decoder core (FFmpeg commit 2da55bf:
// libavcodec/opus/dec.c; ff_exp10 from libavutil/ffmath.h).
// Copyright (c) 2012 Andrew D'Addesio, (c) 2013-2014 Mozilla Corporation;
// LGPL-2.1-or-later (see LICENSE-LGPL).

use std::collections::VecDeque;

use super::celt::Celt;
use super::parse::{parse_packet, Mode, OpusPacket, ParseContext, BANDWIDTH_WIDEBAND};
use super::rc::RangeDecoder;
use super::silk::Silk;
use super::swr::Swr;
use super::tab::{CELT_BAND_END, CELT_WINDOW2};

static SILK_FRAME_DURATION_MS: [usize; 16] = [
    10, 20, 40, 60, 10, 20, 40, 60, 10, 20, 40, 60, 10, 20, 10, 20,
];

/// Silence fed to the resampler when it starts, per bandwidth.
static SILK_RESAMPLE_DELAY: [usize; 5] = [4, 8, 11, 11, 11];

fn get_silk_samplerate(config: usize) -> i64 {
    if config < 4 {
        8000
    } else if config < 8 {
        12000
    } else {
        16000
    }
}

/// `opus_fade`: `out = in2 * window + in1 * (1 - window)`.
fn fade(out: &mut [f32], in1: &[f32], in2: &[f32], window: &[f32], len: usize) {
    for i in 0..len {
        out[i] = (f64::from(in2[i] * window[i]) + f64::from(in1[i]) * (1.0 - f64::from(window[i])))
            as f32;
    }
}

/// Why a packet did not decode (FFmpeg's error return).
#[derive(Debug)]
pub(crate) struct DecodeError(pub(crate) &'static str);

/// `OpusStreamContext`.
struct Stream {
    output_channels: usize,
    decoded_samples: usize,
    packet: OpusPacket,
    silk: Silk,
    celt: Celt,
    swr: Swr,
    celt_delay: [VecDeque<f32>; 2],
    sync_buffer: [VecDeque<f32>; 2],
    silk_buf: Box<[[f32; 960]; 2]>,
    celt_buf: Box<[[f32; 960]; 2]>,
    redundancy_buf: Box<[[f32; 960]; 2]>,
    silk_samplerate: i64,
    delayed_samples: usize,
    redundancy_idx: usize,
}

impl Stream {
    fn new(output_channels: usize, apply_phase_inv: bool) -> Self {
        Self {
            output_channels,
            decoded_samples: 0,
            packet: OpusPacket::default(),
            silk: Silk::new(output_channels),
            celt: Celt::new(output_channels, apply_phase_inv),
            swr: Swr::new(output_channels),
            celt_delay: [VecDeque::new(), VecDeque::new()],
            sync_buffer: [VecDeque::new(), VecDeque::new()],
            silk_buf: Box::new([[0.0; 960]; 2]),
            celt_buf: Box::new([[0.0; 960]; 2]),
            redundancy_buf: Box::new([[0.0; 960]; 2]),
            silk_samplerate: 0,
            delayed_samples: 0,
            redundancy_idx: 0,
        }
    }

    /// `opus_flush_resample`: `nb` samples at `out[c][pos..]`.
    fn flush_resample(
        &mut self,
        out: &mut [Vec<f32>; 2],
        pos: usize,
        nb: usize,
    ) -> Result<(), DecodeError> {
        let celt_size = self.celt_delay[0].len();
        let ret = {
            let (a, b) = out.split_at_mut(1);
            let mut planes: Vec<&mut [f32]> = vec![&mut a[0][pos..], &mut b[0][pos..]];
            planes.truncate(self.output_channels);
            self.swr.convert(&mut planes, nb, None, 0)
        };
        if ret != nb {
            return Err(DecodeError("wrong number of flushed samples"));
        }
        if celt_size != 0 {
            if celt_size != nb {
                return Err(DecodeError("wrong number of CELT delay samples"));
            }
            for c in 0..self.output_channels {
                for k in 0..nb {
                    let v = self.celt_delay[c].pop_front().unwrap_or(0.0);
                    out[c][pos + k] += v * 1.0;
                }
            }
        }
        if self.redundancy_idx != 0 {
            let idx = self.redundancy_idx;
            for c in 0..self.output_channels {
                let cur: Vec<f32> = out[c][pos..pos + 120 - idx].to_vec();
                fade(
                    &mut out[c][pos..],
                    &cur,
                    &self.redundancy_buf[c][120 + idx..],
                    &CELT_WINDOW2[idx..],
                    120 - idx,
                );
            }
            self.redundancy_idx = 0;
        }
        Ok(())
    }

    /// `opus_init_resample`.
    fn init_resample(&mut self) {
        self.swr.init(self.silk_samplerate);
        let delay = [0f32; 16];
        let planes: [&[f32]; 2] = [&delay, &delay];
        let n = SILK_RESAMPLE_DELAY[self.packet.bandwidth];
        self.swr
            .convert(&mut [], 0, Some(&planes[..self.output_channels]), n);
    }

    /// `opus_decode_redundancy`: the redundant frame is the first `size`
    /// bytes of `data` (the rest of the packet follows it).
    fn decode_redundancy(&mut self, data: &[u8], size: usize) -> Result<(), DecodeError> {
        let mut rc = RangeDecoder::new(data, size);
        rc.raw_init(size, size as u32);
        let channels = usize::from(self.packet.stereo) + 1;
        let bw = self.packet.bandwidth;
        let (a, b) = self.redundancy_buf.split_at_mut(1);
        let mut planes: [&mut [f32]; 2] = [&mut a[0][..], &mut b[0][..]];
        self.celt
            .decode_frame(
                &mut rc,
                &mut planes,
                channels,
                240,
                0,
                usize::from(CELT_BAND_END[bw]),
            )
            .map_err(|_| DecodeError("error decoding the redundancy frame"))
    }

    /// `opus_decode_frame`: one frame into `out[c][pos..]`; returns the
    /// samples written.
    fn decode_frame(
        &mut self,
        out: &mut [Vec<f32>; 2],
        pos: usize,
        data: &[u8],
        frame_size: usize,
    ) -> Result<usize, DecodeError> {
        let mut samples = self.packet.frame_duration;
        let mut redundancy = 0u32;
        let mut redundancy_size = 0usize;
        let mut redundancy_pos = 0u32;
        let delayed_samples = self.delayed_samples;
        let mut size = frame_size;
        let mode = self.packet.mode;
        let mut rc = RangeDecoder::new(data, size);
        // Redundancy writes reach `delayed_samples` + 240 past `pos`.
        ensure(
            out,
            pos + self.packet.frame_duration + delayed_samples + 240,
        );

        if mode == Mode::Silk || mode == Mode::Hybrid {
            if !self.swr.is_initialized() {
                self.init_resample();
            }
            let n = self
                .silk
                .decode_superframe(
                    &mut rc,
                    &mut self.silk_buf,
                    self.packet.bandwidth.min(BANDWIDTH_WIDEBAND),
                    usize::from(self.packet.stereo) + 1,
                    SILK_FRAME_DURATION_MS[self.packet.config],
                )
                .ok_or(DecodeError("error decoding a SILK frame"))?;
            let silk = &self.silk_buf;
            let inputs: [&[f32]; 2] = [&silk[0][..], &silk[1][..]];
            let (a, b) = out.split_at_mut(1);
            let mut planes: Vec<&mut [f32]> = vec![&mut a[0][pos..], &mut b[0][pos..]];
            planes.truncate(self.output_channels);
            samples = self.swr.convert(
                &mut planes,
                self.packet.frame_duration,
                Some(&inputs[..self.output_channels]),
                n,
            );
            self.delayed_samples += self.packet.frame_duration - samples;
        } else {
            self.silk.flush();
        }

        let consumed = rc.tell() as usize;
        if mode == Mode::Hybrid && consumed + 37 <= size * 8 {
            redundancy = rc.dec_log(12);
        } else if mode == Mode::Silk && consumed + 17 <= size * 8 {
            redundancy = 1;
        }
        if redundancy != 0 {
            redundancy_pos = rc.dec_log(1);
            redundancy_size = if mode == Mode::Hybrid {
                rc.dec_uint(256) as usize + 2
            } else {
                size.saturating_sub(consumed.div_ceil(8))
            };
            if redundancy_size > size {
                return Err(DecodeError("invalid redundancy frame size"));
            }
            size -= redundancy_size;
            if redundancy_pos != 0 {
                self.decode_redundancy(&data[size..], redundancy_size)?;
                self.celt.flush();
            }
        }

        if mode == Mode::Celt || mode == Mode::Hybrid {
            let mut out_off = [pos, pos];
            let mut celt_output_samples = samples;
            let delay_samples = self.celt_delay[0].len();
            if delay_samples != 0 {
                if mode == Mode::Hybrid {
                    for c in 0..self.output_channels {
                        for k in 0..delay_samples {
                            self.celt_buf[c][k] = self.celt_delay[c].pop_front().unwrap_or(0.0);
                        }
                        for k in 0..delay_samples {
                            out[c][out_off[c] + k] += self.celt_buf[c][k] * 1.0;
                        }
                        out_off[c] += delay_samples;
                    }
                    celt_output_samples = celt_output_samples.saturating_sub(delay_samples);
                } else {
                    for f in &mut self.celt_delay {
                        f.clear();
                    }
                }
            }

            rc.raw_init(size, size as u32);
            let channels = usize::from(self.packet.stereo) + 1;
            let fd = self.packet.frame_duration;
            let start = if mode == Mode::Hybrid { 17 } else { 0 };
            let end = usize::from(CELT_BAND_END[self.packet.bandwidth]);
            if mode == Mode::Celt {
                let (a, b) = out.split_at_mut(1);
                let mut planes: [&mut [f32]; 2] =
                    [&mut a[0][out_off[0]..], &mut b[0][out_off[1]..]];
                self.celt
                    .decode_frame(&mut rc, &mut planes, channels, fd, start, end)
                    .map_err(|_| DecodeError("error decoding a CELT frame"))?;
            } else {
                {
                    let (a, b) = self.celt_buf.split_at_mut(1);
                    let mut planes: [&mut [f32]; 2] = [&mut a[0][..], &mut b[0][..]];
                    self.celt
                        .decode_frame(&mut rc, &mut planes, channels, fd, start, end)
                        .map_err(|_| DecodeError("error decoding a CELT frame"))?;
                }
                let celt_delay = fd.saturating_sub(celt_output_samples);
                for c in 0..self.output_channels {
                    for k in 0..celt_output_samples {
                        out[c][out_off[c] + k] += self.celt_buf[c][k] * 1.0;
                    }
                    for k in 0..celt_delay {
                        self.celt_delay[c].push_back(self.celt_buf[c][celt_output_samples + k]);
                    }
                }
            }
        } else {
            self.celt.flush();
        }

        if self.redundancy_idx != 0 {
            let idx = self.redundancy_idx;
            for c in 0..self.output_channels {
                let cur: Vec<f32> = out[c][pos..pos + 120 - idx].to_vec();
                fade(
                    &mut out[c][pos..],
                    &cur,
                    &self.redundancy_buf[c][120 + idx..],
                    &CELT_WINDOW2[idx..],
                    120 - idx,
                );
            }
            self.redundancy_idx = 0;
        }
        if redundancy != 0 {
            if redundancy_pos == 0 {
                self.celt.flush();
                self.decode_redundancy(&data[size..], redundancy_size)?;
                for c in 0..self.output_channels {
                    let at = (pos + samples + delayed_samples)
                        .checked_sub(120)
                        .ok_or(DecodeError("frame too short"))?;
                    let len = 120usize.saturating_sub(delayed_samples);
                    let cur: Vec<f32> = out[c][at..at + len].to_vec();
                    fade(
                        &mut out[c][at..],
                        &cur,
                        &self.redundancy_buf[c][120..],
                        &CELT_WINDOW2,
                        len,
                    );
                    if delayed_samples != 0 {
                        self.redundancy_idx = 120 - delayed_samples.min(120);
                    }
                }
            } else {
                for c in 0..self.output_channels {
                    let at = pos + delayed_samples;
                    out[c][at..at + 120].copy_from_slice(&self.redundancy_buf[c][..120]);
                    let cur: Vec<f32> = out[c][at + 120..at + 240].to_vec();
                    fade(
                        &mut out[c][at + 120..],
                        &self.redundancy_buf[c][120..240],
                        &cur,
                        &CELT_WINDOW2,
                        120,
                    );
                }
            }
        }
        Ok(samples)
    }

    /// `opus_decode_subpacket`: decodes the packet already parsed into
    /// `self.packet` (or, with `None`, flushes) into `out[c][pos..]`;
    /// returns the samples written.
    fn decode_subpacket(
        &mut self,
        out: &mut [Vec<f32>; 2],
        pos: usize,
        buf: Option<&[u8]>,
    ) -> Result<usize, DecodeError> {
        let mut output_samples = 0usize;
        let mut flush_needed = false;
        let mut cur = pos;
        if self.swr.is_initialized() {
            if buf.is_some() {
                flush_needed =
                    self.packet.mode == Mode::Celt || self.swr.in_rate() != self.silk_samplerate;
            } else {
                flush_needed = self.delayed_samples != 0;
            }
        }
        if buf.is_none() && !flush_needed {
            return Ok(0);
        }
        if flush_needed {
            let nb = self.delayed_samples;
            ensure(out, cur + nb + 240);
            self.flush_resample(out, cur, nb)?;
            cur += nb;
            self.swr.close();
            output_samples += nb;
            self.delayed_samples = 0;
        }
        let Some(buf) = buf else {
            return Ok(output_samples);
        };
        for i in 0..self.packet.frame_count {
            let (off, size) = (self.packet.frame_offset[i], self.packet.frame_size[i]);
            ensure(out, cur + self.packet.frame_duration + 240);
            // The frame and the rest of the packet after it.
            let data = buf
                .get(off..)
                .filter(|d| d.len() >= size)
                .ok_or(DecodeError("frame outside the packet"))?;
            let samples = match self.decode_frame(out, cur, data, size) {
                Ok(s) => s,
                Err(_) => {
                    // FFmpeg (without AV_EF_EXPLODE): a frame that fails
                    // to decode is silence.
                    let fd = self.packet.frame_duration;
                    for c in out.iter_mut().take(self.output_channels) {
                        c[cur..cur + fd].fill(0.0);
                    }
                    fd
                }
            };
            output_samples += samples;
            cur += samples;
        }
        Ok(output_samples)
    }
}

fn ensure(out: &mut [Vec<f32>; 2], len: usize) {
    for c in out.iter_mut() {
        if c.len() < len {
            c.resize(len, 0.0);
        }
    }
}

/// `OpusContext`: the multistream decoder.
pub(crate) struct OpusContext {
    streams: Vec<Stream>,
    gain: f32,
    gain_i: i16,
    pub(crate) p: ParseContext,
}

/// One decoded frame: planar float, one plane per output channel.
pub(crate) struct DecodedFrame {
    pub(crate) planes: Vec<Vec<f32>>,
    pub(crate) samples: usize,
}

impl OpusContext {
    /// `opus_decode_init`.
    pub(crate) fn new(p: ParseContext) -> Self {
        let streams = (0..p.nb_streams)
            .map(|i| Stream::new(if i < p.nb_stereo_streams { 2 } else { 1 }, true))
            .collect();
        let gain = if p.gain_i != 0 {
            2f64.powf(std::f64::consts::LOG2_10 * (f64::from(p.gain_i) / (20.0 * 256.0))) as f32
        } else {
            0.0
        };
        Self {
            streams,
            gain,
            gain_i: p.gain_i,
            p,
        }
    }

    pub(crate) fn channels(&self) -> usize {
        self.p.channel_maps.len()
    }

    /// `opus_decode_flush`.
    pub(crate) fn flush(&mut self) {
        for s in &mut self.streams {
            s.packet = OpusPacket::default();
            s.delayed_samples = 0;
            for f in &mut s.celt_delay {
                f.clear();
            }
            s.swr.close();
            for f in &mut s.sync_buffer {
                f.clear();
            }
            s.silk.flush();
            s.celt.flush();
        }
    }

    /// `opus_decode_packet`: `None` drains. Returns the frame, if any.
    pub(crate) fn decode_packet(
        &mut self,
        buf: Option<&[u8]>,
    ) -> Result<Option<DecodedFrame>, DecodeError> {
        let nb_streams = self.streams.len();
        let mut coded_samples = 0usize;
        let mut delayed_samples = 0usize;
        for s in &self.streams {
            delayed_samples = delayed_samples.max(s.delayed_samples + s.sync_buffer[0].len());
        }
        if let Some(data) = buf {
            let s0 = &mut self.streams[0];
            parse_packet(&mut s0.packet, data, nb_streams > 1)
                .map_err(|_| DecodeError("error parsing the packet header"))?;
            coded_samples += s0.packet.frame_count * s0.packet.frame_duration;
            s0.silk_samplerate = get_silk_samplerate(s0.packet.config);
        }
        let nb_samples = coded_samples + delayed_samples;
        if nb_samples == 0 {
            return Ok(None);
        }
        let channels = self.channels();

        // Which output channel each stream channel writes (the last map
        // naming it, as FFmpeg assigns them in order; silent maps name
        // stream 0, channel 0, as FFmpeg's zeroed maps do).
        let mut targets = vec![[None::<usize>; 2]; nb_streams];
        for (i, map) in self.p.channel_maps.iter().enumerate() {
            if map.copy.is_none() {
                if let Some(t) = targets.get_mut(map.stream_idx) {
                    t[map.channel_idx.min(1)] = Some(i);
                }
            }
        }

        let mut outs: Vec<[Vec<f32>; 2]> = Vec::with_capacity(nb_streams);
        let mut sync_sizes = Vec::with_capacity(nb_streams);
        for s in &mut self.streams {
            let mut o = [vec![0f32; nb_samples + 240], vec![0f32; nb_samples + 240]];
            let sync = s.sync_buffer[0].len();
            for c in 0..2 {
                for k in 0..sync {
                    o[c][k] = s.sync_buffer[c].pop_front().unwrap_or(0.0);
                }
            }
            outs.push(o);
            sync_sizes.push(sync);
        }

        let mut decoded_samples = usize::MAX;
        let mut offset = 0usize;
        for i in 0..nb_streams {
            if i != 0 {
                if let Some(data) = buf {
                    let rest = data.get(offset..).unwrap_or(&[]);
                    let s = &mut self.streams[i];
                    parse_packet(&mut s.packet, rest, i != nb_streams - 1)
                        .map_err(|_| DecodeError("error parsing the packet header"))?;
                    if coded_samples != s.packet.frame_count * s.packet.frame_duration {
                        return Err(DecodeError("mismatching coded sample count in a substream"));
                    }
                    s.silk_samplerate = get_silk_samplerate(s.packet.config);
                }
            }
            let sub = buf.map(|data| data.get(offset..).unwrap_or(&[]));
            let s = &mut self.streams[i];
            let ret = s.decode_subpacket(&mut outs[i], sync_sizes[i], sub)?;
            s.decoded_samples = ret;
            decoded_samples = decoded_samples.min(ret);
            if buf.is_some() {
                offset += s.packet.packet_size;
            }
        }

        for (i, s) in self.streams.iter_mut().enumerate() {
            let extra = s.decoded_samples - decoded_samples;
            if extra != 0 {
                let from = sync_sizes[i] + decoded_samples;
                for c in 0..2 {
                    // FFmpeg buffers an unmapped channel from output 0.
                    for k in 0..extra {
                        let v = outs[i][c].get(from + k).copied().unwrap_or(0.0);
                        s.sync_buffer[c].push_back(v);
                    }
                }
            }
        }

        let mut planes: Vec<Vec<f32>> = vec![Vec::new(); channels];
        for (si, t) in targets.iter().enumerate() {
            for c in 0..2 {
                if let Some(ch) = t[c] {
                    planes[ch] = outs[si][c][..decoded_samples].to_vec();
                }
            }
        }
        for i in 0..channels {
            let map = self.p.channel_maps[i];
            if let Some(src) = map.copy {
                planes[i] = planes[src].clone();
            } else if map.silence {
                planes[i] = vec![0.0; decoded_samples];
            }
            if planes[i].len() != decoded_samples {
                planes[i].resize(decoded_samples, 0.0);
            }
            if self.gain_i != 0 && decoded_samples > 0 {
                for v in &mut planes[i] {
                    *v *= self.gain;
                }
            }
        }
        if decoded_samples == 0 {
            return Ok(None);
        }
        Ok(Some(DecodedFrame {
            planes,
            samples: decoded_samples,
        }))
    }
}
