//! Untrusted input through the registered decoder (`make_decoder`, the
//! FFmpeg port): truncated, bit-flipped and byte-substituted copies of real
//! packets, and broken `OpusHead`s, must never panic, and what comes out
//! stays within what an Opus packet can hold (120 ms) plus the resampler's
//! delayed samples. Deterministic: fixed seed.

use oxideav_core::{CodecId, CodecParameters, Error, Frame, Packet, Rational, TimeBase};

fn ogg_packets(data: &[u8]) -> Vec<Vec<u8>> {
    let mut off = 0usize;
    let mut packets = Vec::new();
    let mut cur = Vec::new();
    while off + 27 <= data.len() {
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

/// xorshift64*.
struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x >> 12;
        x ^= x << 25;
        x ^= x >> 27;
        self.0 = x;
        x.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }
    fn below(&mut self, n: usize) -> usize {
        (self.next() % n.max(1) as u64) as usize
    }
}

fn mutate(rng: &mut Rng, data: &[u8]) -> Vec<u8> {
    let mut m = data.to_vec();
    if m.is_empty() {
        return m;
    }
    match rng.below(3) {
        0 => m.truncate(rng.below(m.len())),
        1 => {
            for _ in 0..1 + rng.below(8) {
                let at = rng.below(m.len());
                m[at] ^= 1 << rng.below(8);
            }
        }
        _ => {
            let at = rng.below(m.len());
            m[at] = rng.next() as u8;
        }
    }
    m
}

/// Feeds `packets` (one in four mutated) through a decoder made from
/// `head`, then flushes, checking every frame's size.
fn feed(head: &[u8], packets: &[Vec<u8>], rng: &mut Rng) {
    let mut params = CodecParameters::audio(CodecId::new("opus"));
    params.extradata = head.to_vec();
    let Ok(mut dec) = oxideav_opus::make_decoder(&params) else {
        return;
    };
    let channels = dec
        .output_audio_format()
        .map_or(0, |f| usize::from(f.channels));
    let tb = TimeBase(Rational::new(1, 48_000));
    let drain = |dec: &mut Box<dyn oxideav_core::Decoder>| {
        for _ in 0..64 {
            match dec.receive_frame() {
                Ok(Frame::Audio(f)) => {
                    assert!(
                        f.samples <= 5760 + 960,
                        "{} samples from one packet",
                        f.samples
                    );
                    assert_eq!(f.data.len(), channels);
                    assert!(f.data.iter().all(|p| p.len() == 4 * f.samples as usize));
                }
                Ok(_) => panic!("audio frames only"),
                Err(Error::NeedMore) | Err(Error::Eof) => return,
                Err(_) => return,
            }
        }
        panic!("receive_frame never ran dry");
    };
    for p in packets {
        let data = if rng.below(4) == 0 {
            mutate(rng, p)
        } else {
            p.clone()
        };
        let _ = dec.send_packet(&Packet::new(0, tb, data));
        drain(&mut dec);
    }
    let _ = dec.flush();
    drain(&mut dec);
    let _ = dec.reset();
}

#[test]
fn mutated_packets_never_panic() {
    let streams: [&[u8]; 6] = [
        include_bytes!("fixtures/celt-7.1-libopus.opus"),
        include_bytes!("fixtures/multistream-5.1.opus"),
        include_bytes!("fixtures/mode-switching.opus"),
        include_bytes!("fixtures/silk-wb-stereo-20kbps.opus"),
        include_bytes!("fixtures/fec-on.opus"),
        include_bytes!("fixtures/celt-2.5ms-low-latency.opus"),
    ];
    let mut rng = Rng(0x0BAD_5EED_0905_2026);
    let mut runs = 0;
    for stream in streams {
        let packets = ogg_packets(stream);
        let head = &packets[0];
        // Keep each run short: a window of packets from a random start.
        for _ in 0..400 {
            let start = 2 + rng.below(packets.len() - 2);
            let end = (start + 1 + rng.below(12)).min(packets.len());
            feed(head, &packets[start..end], &mut rng);
            runs += 1;
        }
    }
    assert!(runs >= 2000, "{runs} runs");
}

#[test]
fn broken_opus_heads_never_panic() {
    let packets = ogg_packets(include_bytes!("fixtures/celt-7.1-libopus.opus"));
    let mut rng = Rng(0x0BAD_4EAD_0905_2026);
    for _ in 0..2000 {
        let head = mutate(&mut rng, &packets[0]);
        feed(&head, &packets[2..4], &mut rng);
    }
}
