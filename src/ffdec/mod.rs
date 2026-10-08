//! FFmpeg's Opus decoder (FFmpeg commit 2da55bf, `libavcodec/opus/dec.c`
//! with its SILK, CELT, PVQ and range-coder files, the `libswresample`
//! resampler it runs SILK through, and its tables), ported to Rust. This is
//! the decoder [`crate::make_decoder`] returns and the registry installs:
//! its output is FFmpeg's `fltp` output for the same packets, frame for
//! frame and channel for channel.
//!
//! What it reproduces, as FFmpeg 2da55bf does it:
//! * FFmpeg's channel order: mapping family 1 (Vorbis order in the
//!   `OpusHead`) comes out in FFmpeg's native layout order (`FL FR FC LFE
//!   BL BR SL SR` for 7.1), as `ff_opus_parse_extradata` reorders it;
//! * FFmpeg's float SILK decoder and its SILK-to-48 kHz path through
//!   libswresample (filter size 16, Kaiser window, exact rational phases),
//!   including the start-up silence, the delayed samples and their flush
//!   on a switch to CELT or at the end of the stream;
//! * the CELT decoder, hybrid delay buffering, redundancy frames and
//!   multistream synchronisation buffers of `dec.c`;
//! * the `OpusHead` output gain.
//!
//! Beyond FFmpeg's decoder, as this crate's adapter always did: the
//! `OpusHead` pre-skip is dropped from the start of the output (FFmpeg's
//! demuxer and decode loop do that; a consumer that trims itself clears
//! the field first).
//!
//! The RFC 6716 decoder of this crate stays available behind
//! [`crate::make_native_decoder`].
//!
//! Copyright (c) the FFmpeg developers and the authors named in each file;
//! LGPL-2.1-or-later (see LICENSE-LGPL).

// The ported loops keep FFmpeg's index structure (several arrays indexed by
// the same counter, as in the C), which is easier to check against it.
#[allow(clippy::needless_range_loop)]
mod celt;
#[allow(clippy::needless_range_loop)]
mod dec;
mod mdct;
mod parse;
#[allow(clippy::needless_range_loop)]
mod pvq;
mod rc;
#[allow(clippy::needless_range_loop)]
mod silk;
// av_bessel_i0's coefficients, as FFmpeg writes them.
#[allow(clippy::excessive_precision)]
mod swr;
#[allow(clippy::excessive_precision, clippy::unreadable_literal)]
mod tab;

use std::collections::VecDeque;

use oxideav_core::{
    AudioFormat, AudioFrame, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Result,
    SampleFormat,
};

use dec::OpusContext;

/// The most output channels the decoder accepts (untrusted input bound).
const MAX_CHANNELS: usize = 64;

/// The Opus decoder: FFmpeg's, frame for frame.
pub(crate) struct FfOpusDecoder {
    codec_id: CodecId,
    ctx: OpusContext,
    format: AudioFormat,
    /// Leading samples still to drop (the `OpusHead` pre-skip).
    pre_skip: usize,
    queue: VecDeque<AudioFrame>,
    eof: bool,
}

impl FfOpusDecoder {
    pub(crate) fn new(params: &CodecParameters) -> Result<Self> {
        let extradata = if params.extradata.is_empty() {
            None
        } else {
            Some(&params.extradata[..])
        };
        let p = parse::parse_extradata(extradata, usize::from(params.channels.unwrap_or(0)))
            .map_err(|e| Error::invalid(format!("opus: {e}")))?;
        let channels = p.channel_maps.len();
        if channels > MAX_CHANNELS {
            return Err(Error::unsupported(format!(
                "opus: {channels} channels (at most {MAX_CHANNELS})"
            )));
        }
        let pre_skip = usize::from(p.pre_skip);
        Ok(Self {
            codec_id: params.codec_id.clone(),
            ctx: OpusContext::new(p),
            format: AudioFormat {
                sample_format: SampleFormat::F32P,
                sample_rate: 48_000,
                channels: channels as u16,
            },
            pre_skip,
            queue: VecDeque::new(),
            eof: false,
        })
    }

    /// Queues the frame the context decoded last (`samples` per channel),
    /// less what is left of the pre-skip.
    fn push(&mut self, samples: usize, pts: Option<i64>) {
        let skip = self.pre_skip.min(samples);
        self.pre_skip -= skip;
        let kept = samples - skip;
        if kept == 0 {
            return;
        }
        let data = (0..usize::from(self.format.channels))
            .map(|ch| {
                let mut plane = Vec::with_capacity(4 * kept);
                self.ctx.write_channel(ch, skip, kept, &mut plane);
                plane
            })
            .collect();
        self.queue.push_back(AudioFrame {
            samples: kept as u32,
            pts,
            data,
        });
    }
}

impl Decoder for FfOpusDecoder {
    fn codec_id(&self) -> &CodecId {
        &self.codec_id
    }

    fn send_packet(&mut self, packet: &Packet) -> Result<()> {
        if packet.data.is_empty() {
            return Ok(());
        }
        match self.ctx.decode_packet(Some(&packet.data)) {
            Ok(Some(samples)) => {
                self.push(samples, packet.pts);
                Ok(())
            }
            Ok(None) => Ok(()),
            Err(e) => Err(Error::invalid(format!("opus: {}", e.0))),
        }
    }

    fn receive_frame(&mut self) -> Result<Frame> {
        match self.queue.pop_front() {
            Some(frame) => Ok(Frame::Audio(frame)),
            None if self.eof => Err(Error::Eof),
            None => Err(Error::NeedMore),
        }
    }

    /// End of stream: drains what the decoder holds back (the resampler's
    /// delayed SILK samples), as FFmpeg's decode loop does at EOF.
    fn flush(&mut self) -> Result<()> {
        if !self.eof {
            // Each drain call outputs the samples still delayed; stop when
            // one outputs nothing.
            for _ in 0..4 {
                match self.ctx.decode_packet(None) {
                    Ok(Some(samples)) => self.push(samples, None),
                    _ => break,
                }
            }
        }
        self.eof = true;
        Ok(())
    }

    /// `opus_decode_flush`: a decoder ready for a new position. The
    /// pre-skip applies once, at the start of the stream.
    fn reset(&mut self) -> Result<()> {
        self.ctx.flush();
        self.queue.clear();
        self.eof = false;
        Ok(())
    }

    /// Planar float at 48 kHz with the `OpusHead`'s channel count, in
    /// FFmpeg's channel order; known from the start.
    fn output_audio_format(&self) -> Option<AudioFormat> {
        Some(self.format)
    }
}

/// The registry factory: [`FfOpusDecoder`] for the stream's
/// [`CodecParameters`] (the `OpusHead` in `extradata`, or 1-2 channels
/// without one).
pub(crate) fn make_decoder(params: &CodecParameters) -> Result<Box<dyn Decoder>> {
    Ok(Box::new(FfOpusDecoder::new(params)?))
}
