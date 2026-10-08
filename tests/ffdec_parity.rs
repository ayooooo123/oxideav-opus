//! The registered decoder (`make_decoder`, FFmpeg 2da55bf's decoder ported
//! in `src/ffdec`) against FFmpeg's own output for the same packets.
//!
//! Each `<name>.ffmpeg-2da55bf.f32` holds what FFmpeg 2da55bf decodes from
//! `<name>.opus` with nothing trimmed (`ffmpeg -flags2 +skip_manual -i
//! <name>.opus -f f32le -`): interleaved float in FFmpeg's channel order,
//! cut to its first samples where the stream is long. The decoder gets the
//! Ogg packets with the `OpusHead` pre-skip cleared (so it trims nothing
//! either) and is flushed at the end, as FFmpeg drains its decoder.

use oxideav_core::{
    AudioFormat, CodecId, CodecParameters, Decoder, Error, Frame, Packet, Rational, SampleFormat,
    TimeBase,
};

/// Ogg packets of an Ogg-Opus stream (RFC 3533 page walk).
fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut off = 0usize;
    let mut packets = Vec::new();
    let mut cur = Vec::new();
    while off + 27 <= data.len() {
        assert_eq!(&data[off..off + 4], b"OggS");
        let nseg = data[off + 26] as usize;
        let segtab = &data[off + 27..off + 27 + nseg];
        let mut p = off + 27 + nseg;
        for &s in segtab {
            cur.extend_from_slice(&data[p..p + s as usize]);
            p += s as usize;
            if s < 255 {
                packets.push(std::mem::take(&mut cur));
            }
        }
        off = p;
    }
    packets
}

/// Decodes every audio packet of `stream` and flushes: the reported layout
/// and the interleaved samples.
fn decode(stream: &[u8]) -> (AudioFormat, Vec<f32>) {
    let packets = ogg_packets(stream);
    let mut head = packets[0].clone();
    head[10..12].fill(0);
    let mut params = CodecParameters::audio(CodecId::new("opus"));
    params.extradata = head;
    let mut dec = oxideav_opus::make_decoder(&params).expect("decoder");
    let format = dec
        .output_audio_format()
        .expect("the layout is known from the OpusHead");
    let tb = TimeBase(Rational::new(1, 48_000));
    let mut out = Vec::new();
    let take = |dec: &mut Box<dyn Decoder>, out: &mut Vec<f32>| loop {
        match dec.receive_frame() {
            Ok(Frame::Audio(f)) => {
                assert_eq!(dec.output_audio_format(), Some(format));
                assert_eq!(f.data.len(), usize::from(format.channels));
                for i in 0..f.samples as usize {
                    for plane in &f.data {
                        out.push(f32::from_le_bytes(
                            plane[4 * i..4 * i + 4].try_into().unwrap(),
                        ));
                    }
                }
            }
            Ok(_) => panic!("audio frames only"),
            Err(Error::NeedMore) | Err(Error::Eof) => break,
            Err(e) => panic!("receive_frame: {e}"),
        }
    };
    for p in &packets[2..] {
        dec.send_packet(&Packet::new(0, tb, p.clone()))
            .expect("decode");
        take(&mut dec, &mut out);
    }
    dec.flush().expect("flush");
    take(&mut dec, &mut out);
    (format, out)
}

fn reference(bytes: &[u8]) -> Vec<f32> {
    bytes
        .chunks_exact(4)
        .map(|b| f32::from_le_bytes(b.try_into().unwrap()))
        .collect()
}

fn snr_db(want: &[f32], got: &[f32]) -> f64 {
    let (mut sig, mut err) = (0f64, 0f64);
    for (w, g) in want.iter().zip(got) {
        sig += f64::from(*w) * f64::from(*w);
        err += (f64::from(*w) - f64::from(*g)).powi(2);
    }
    if err == 0.0 {
        f64::INFINITY
    } else {
        10.0 * (sig / err).log10()
    }
}

/// Decodes `stream`, checks its layout and its total length (FFmpeg's, in
/// samples per channel), and returns the SNR of its leading samples against
/// FFmpeg's (`want` may be a prefix of FFmpeg's output).
fn check(stream: &[u8], want: &[u8], channels: u16, total: usize) -> f64 {
    let (format, got) = decode(stream);
    assert_eq!(
        format,
        AudioFormat {
            sample_format: SampleFormat::F32P,
            sample_rate: 48_000,
            channels
        }
    );
    assert_eq!(
        got.len(),
        total * usize::from(channels),
        "samples per channel"
    );
    let want = reference(want);
    assert!(want.len() <= got.len());
    snr_db(&want, &got[..want.len()])
}

/// libopus 7.1 (mapping family 1, 5 streams, 3 coupled), CELT: the eight
/// channels come out in FFmpeg's order (FL FR FC LFE BL BR SL SR), not the
/// `OpusHead`'s Vorbis order.
#[test]
fn surround_7_1_matches_ffmpeg_in_ffmpeg_channel_order() {
    let snr = check(
        include_bytes!("fixtures/celt-7.1-libopus.opus"),
        include_bytes!("fixtures/celt-7.1-libopus.ffmpeg-2da55bf.f32"),
        8,
        5760,
    );
    assert!(snr >= 120.0, "7.1 vs FFmpeg: {snr:.1} dB");
}

/// The 5.1 fixture (family 1, 4 streams, 2 coupled), first 0.1 s.
#[test]
fn surround_5_1_matches_ffmpeg() {
    let snr = check(
        include_bytes!("fixtures/multistream-5.1.opus"),
        include_bytes!("fixtures/multistream-5.1.ffmpeg-2da55bf.f32"),
        6,
        48_960,
    );
    assert!(snr >= 120.0, "5.1 vs FFmpeg: {snr:.1} dB");
}

/// Hybrid and CELT frames with mode switches: the SILK resampler's delay,
/// its flush, the CELT delay buffer and redundancy frames.
#[test]
fn mode_switching_matches_ffmpeg() {
    let snr = check(
        include_bytes!("fixtures/mode-switching.opus"),
        include_bytes!("fixtures/mode-switching.ffmpeg-2da55bf.f32"),
        1,
        72_960,
    );
    assert!(snr >= 120.0, "mode switching vs FFmpeg: {snr:.1} dB");
}

/// Stereo SILK (WB, mid/side): FFmpeg's float SILK and its swresample
/// path, which differ from the RFC 6716 reference decoder; first 0.25 s.
#[test]
fn stereo_silk_matches_ffmpeg() {
    let snr = check(
        include_bytes!("fixtures/silk-wb-stereo-20kbps.opus"),
        include_bytes!("fixtures/silk-wb-stereo-20kbps.ffmpeg-2da55bf.f32"),
        2,
        72_960,
    );
    assert!(snr >= 120.0, "stereo SILK vs FFmpeg: {snr:.1} dB");
}
