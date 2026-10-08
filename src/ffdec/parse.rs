// Port of FFmpeg's Opus packet and extradata parsing (FFmpeg commit 2da55bf:
// libavcodec/opus/parse.c, parse.h, opus.h; the Vorbis channel order from
// libavcodec/vorbis_data.c).
// Copyright (c) 2012 Andrew D'Addesio, (c) 2013-2014 Mozilla Corporation,
// (c) 2005 Denes Balatoni and the FFmpeg developers (vorbis_data.c);
// LGPL-2.1-or-later (see LICENSE-LGPL).

use super::tab::OPUS_FRAME_DURATION;

pub(crate) const OPUS_MAX_FRAME_SIZE: usize = 1275;
pub(crate) const OPUS_MAX_FRAMES: usize = 48;
pub(crate) const OPUS_MAX_PACKET_DUR: usize = 5760;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Mode {
    #[default]
    Silk,
    Hybrid,
    Celt,
}

/// `OpusBandwidth`: 0 narrowband .. 4 fullband.
pub(crate) const BANDWIDTH_NARROWBAND: usize = 0;
pub(crate) const BANDWIDTH_WIDEBAND: usize = 2;
pub(crate) const BANDWIDTH_SUPERWIDEBAND: usize = 3;

/// `OpusPacket`.
#[derive(Clone, Debug)]
pub(crate) struct OpusPacket {
    pub(crate) packet_size: usize,
    pub(crate) stereo: bool,
    pub(crate) config: usize,
    pub(crate) frame_count: usize,
    pub(crate) frame_offset: [usize; OPUS_MAX_FRAMES],
    pub(crate) frame_size: [usize; OPUS_MAX_FRAMES],
    pub(crate) frame_duration: usize,
    pub(crate) mode: Mode,
    pub(crate) bandwidth: usize,
}

impl Default for OpusPacket {
    fn default() -> Self {
        Self {
            packet_size: 0,
            stereo: false,
            config: 0,
            frame_count: 0,
            frame_offset: [0; OPUS_MAX_FRAMES],
            frame_size: [0; OPUS_MAX_FRAMES],
            frame_duration: 0,
            mode: Mode::Silk,
            bandwidth: 0,
        }
    }
}

/// `xiph_lacing_16bit`.
fn lacing_16bit(buf: &[u8], ptr: &mut usize, end: usize) -> Option<usize> {
    if *ptr >= end {
        return None;
    }
    let mut val = usize::from(buf[*ptr]);
    *ptr += 1;
    if val >= 252 {
        if *ptr >= end {
            return None;
        }
        val += 4 * usize::from(buf[*ptr]);
        *ptr += 1;
    }
    Some(val)
}

/// `xiph_lacing_full`.
fn lacing_full(buf: &[u8], ptr: &mut usize, end: usize) -> Option<usize> {
    let mut val = 0usize;
    loop {
        if *ptr >= end || val > i32::MAX as usize - 254 {
            return None;
        }
        let next = usize::from(buf[*ptr]);
        *ptr += 1;
        val += next;
        if next < 255 {
            break;
        }
        val -= 1;
    }
    Some(val)
}

/// `ff_opus_parse_packet`. On failure the packet is reset, as FFmpeg's
/// `memset` does.
pub(crate) fn parse_packet(
    pkt: &mut OpusPacket,
    buf: &[u8],
    self_delimiting: bool,
) -> Result<(), ()> {
    let r = parse_inner(pkt, buf, self_delimiting);
    if r.is_err() {
        *pkt = OpusPacket::default();
    }
    r
}

fn parse_inner(pkt: &mut OpusPacket, buf: &[u8], self_delimiting: bool) -> Result<(), ()> {
    let mut buf_size = buf.len();
    let mut end = buf.len();
    let mut ptr = 0usize;
    let mut padding = 0usize;
    if buf_size < 1 {
        return Err(());
    }
    let toc = buf[ptr];
    ptr += 1;
    let code = toc & 3;
    pkt.stereo = (toc >> 2) & 1 != 0;
    pkt.config = usize::from(toc >> 3);

    if code >= 2 && buf_size < 2 {
        return Err(());
    }

    match code {
        0 => {
            pkt.frame_count = 1;
            if self_delimiting {
                let len = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
                if len > end - ptr {
                    return Err(());
                }
                end = ptr + len;
                buf_size = end;
            }
            let frame_bytes = end - ptr;
            if frame_bytes > OPUS_MAX_FRAME_SIZE {
                return Err(());
            }
            pkt.frame_offset[0] = ptr;
            pkt.frame_size[0] = frame_bytes;
        }
        1 => {
            pkt.frame_count = 2;
            if self_delimiting {
                let len = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
                if 2 * len > end - ptr {
                    return Err(());
                }
                end = ptr + 2 * len;
                buf_size = end;
            }
            let frame_bytes = end - ptr;
            if frame_bytes & 1 != 0 || frame_bytes >> 1 > OPUS_MAX_FRAME_SIZE {
                return Err(());
            }
            pkt.frame_offset[0] = ptr;
            pkt.frame_size[0] = frame_bytes >> 1;
            pkt.frame_offset[1] = pkt.frame_offset[0] + pkt.frame_size[0];
            pkt.frame_size[1] = frame_bytes >> 1;
        }
        2 => {
            pkt.frame_count = 2;
            let first = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
            if self_delimiting {
                let len = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
                if len + first > end - ptr {
                    return Err(());
                }
                end = ptr + first + len;
                buf_size = end;
            }
            pkt.frame_offset[0] = ptr;
            pkt.frame_size[0] = first;
            let second = (end - ptr).checked_sub(first).ok_or(())?;
            if second > OPUS_MAX_FRAME_SIZE {
                return Err(());
            }
            pkt.frame_offset[1] = pkt.frame_offset[0] + pkt.frame_size[0];
            pkt.frame_size[1] = second;
        }
        _ => {
            let i = buf[ptr];
            ptr += 1;
            pkt.frame_count = usize::from(i & 0x3f);
            let has_padding = (i >> 6) & 1 != 0;
            let vbr = (i >> 7) & 1 != 0;
            if pkt.frame_count == 0 || pkt.frame_count > OPUS_MAX_FRAMES {
                return Err(());
            }
            if has_padding {
                padding = lacing_full(buf, &mut ptr, end).ok_or(())?;
            }
            if vbr {
                let mut total_bytes = 0usize;
                for k in 0..pkt.frame_count - 1 {
                    let fb = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
                    pkt.frame_size[k] = fb;
                    total_bytes += fb;
                }
                if self_delimiting {
                    let len = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
                    if len + total_bytes + padding > end - ptr {
                        return Err(());
                    }
                    end = ptr + total_bytes + len + padding;
                    buf_size = end;
                }
                let frame_bytes = (end - ptr).checked_sub(padding).ok_or(())?;
                if total_bytes > frame_bytes {
                    return Err(());
                }
                pkt.frame_offset[0] = ptr;
                for k in 1..pkt.frame_count {
                    pkt.frame_offset[k] = pkt.frame_offset[k - 1] + pkt.frame_size[k - 1];
                }
                pkt.frame_size[pkt.frame_count - 1] = frame_bytes - total_bytes;
            } else {
                let frame_bytes;
                if self_delimiting {
                    let fb = lacing_16bit(buf, &mut ptr, end).ok_or(())?;
                    if pkt.frame_count * fb + padding > end - ptr {
                        return Err(());
                    }
                    end = ptr + pkt.frame_count * fb + padding;
                    buf_size = end;
                    frame_bytes = fb;
                } else {
                    let fb = (end - ptr).checked_sub(padding).ok_or(())?;
                    if fb % pkt.frame_count != 0 || fb / pkt.frame_count > OPUS_MAX_FRAME_SIZE {
                        return Err(());
                    }
                    frame_bytes = fb / pkt.frame_count;
                }
                pkt.frame_offset[0] = ptr;
                pkt.frame_size[0] = frame_bytes;
                for k in 1..pkt.frame_count {
                    pkt.frame_offset[k] = pkt.frame_offset[k - 1] + pkt.frame_size[k - 1];
                    pkt.frame_size[k] = frame_bytes;
                }
            }
        }
    }

    pkt.packet_size = buf_size;
    pkt.frame_duration = usize::from(OPUS_FRAME_DURATION[pkt.config]);
    if pkt.frame_duration * pkt.frame_count > OPUS_MAX_PACKET_DUR {
        return Err(());
    }
    if pkt.config < 12 {
        pkt.mode = Mode::Silk;
        pkt.bandwidth = pkt.config >> 2;
    } else if pkt.config < 16 {
        pkt.mode = Mode::Hybrid;
        pkt.bandwidth = BANDWIDTH_SUPERWIDEBAND + usize::from(pkt.config >= 14);
    } else {
        pkt.mode = Mode::Celt;
        pkt.bandwidth = (pkt.config - 16) >> 2;
        if pkt.bandwidth != 0 {
            pkt.bandwidth += 1;
        }
    }
    Ok(())
}

/// `ff_vorbis_channel_layout_offsets`: Vorbis (mapping family 1) channel
/// `vorbis[i]` feeds FFmpeg's native channel `i`.
static VORBIS_CHANNEL_LAYOUT_OFFSETS: [[u8; 8]; 8] = [
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 0, 0, 0, 0, 0, 0],
    [0, 2, 1, 0, 0, 0, 0, 0],
    [0, 1, 2, 3, 0, 0, 0, 0],
    [0, 2, 1, 3, 4, 0, 0, 0],
    [0, 2, 1, 5, 3, 4, 0, 0],
    [0, 2, 1, 6, 5, 3, 4, 0],
    [0, 2, 1, 7, 5, 6, 3, 4],
];

/// `ChannelMap`: where output channel `i` comes from.
#[derive(Clone, Copy, Debug, Default)]
pub(crate) struct ChannelMap {
    pub(crate) stream_idx: usize,
    pub(crate) channel_idx: usize,
    /// A copy of output channel `copy_idx`.
    pub(crate) copy: Option<usize>,
    pub(crate) silence: bool,
}

/// `OpusParseContext` plus what `ff_opus_parse_extradata` sets on the codec
/// context.
#[derive(Clone, Debug)]
pub(crate) struct ParseContext {
    pub(crate) nb_streams: usize,
    pub(crate) nb_stereo_streams: usize,
    pub(crate) gain_i: i16,
    pub(crate) channel_maps: Vec<ChannelMap>,
    /// `avctx->delay`: the pre-skip, in 48 kHz samples.
    pub(crate) pre_skip: u16,
}

/// `ff_opus_parse_extradata`. `extradata` is the `OpusHead`, or `None` for a
/// stream without one (then `channels`, 1 or 2, comes from the container).
pub(crate) fn parse_extradata(
    extradata: Option<&[u8]>,
    channels: usize,
) -> Result<ParseContext, String> {
    const DEFAULT: [u8; 30] = {
        let mut d = [0u8; 30];
        d[0] = b'O';
        d[1] = b'p';
        d[2] = b'u';
        d[3] = b's';
        d[4] = b'H';
        d[5] = b'e';
        d[6] = b'a';
        d[7] = b'd';
        d[8] = 1;
        d
    };
    let has_extradata = extradata.is_some();
    let ed: &[u8] = match extradata {
        Some(ed) => ed,
        None => {
            if channels > 2 {
                return Err("multichannel configuration without extradata".into());
            }
            &DEFAULT
        }
    };
    if ed.len() < 19 {
        return Err(format!("invalid extradata size: {}", ed.len()));
    }
    let version = ed[8];
    if version > 15 {
        return Err(format!("extradata version {version} is not supported"));
    }
    let pre_skip = u16::from_le_bytes([ed[10], ed[11]]);
    let channels = if has_extradata {
        usize::from(ed[9])
    } else if channels == 1 {
        1
    } else {
        2
    };
    if channels == 0 {
        return Err("zero channel count specified in the extradata".into());
    }
    let gain_i = i16::from_le_bytes([ed[16], ed[17]]);
    let map_type = ed[18];
    let (streams, stereo_streams, channel_map, vorbis): (usize, usize, &[u8], bool);
    if map_type == 0 {
        if channels > 2 {
            return Err("channel mapping 0 is only specified for up to 2 channels".into());
        }
        streams = 1;
        stereo_streams = channels - 1;
        channel_map = &[0, 1];
        vorbis = false;
    } else if map_type == 1 || map_type == 2 || map_type == 255 {
        if ed.len() < 21 + channels {
            return Err(format!("invalid extradata size: {}", ed.len()));
        }
        streams = usize::from(ed[19]);
        stereo_streams = usize::from(ed[20]);
        if streams == 0 || stereo_streams > streams || streams + stereo_streams > 255 {
            return Err(format!(
                "invalid stream/stereo stream count: {streams}/{stereo_streams}"
            ));
        }
        if map_type == 1 {
            if channels > 8 {
                return Err("channel mapping 1 is only specified for up to 8 channels".into());
            }
            vorbis = true;
        } else {
            if map_type == 2 {
                let order = ff_isqrt(channels) - 1;
                let square = (order + 1) * (order + 1);
                if channels != square && channels != square + 2 {
                    return Err(
                        "channel mapping 2 needs (n + 1)^2 or (n + 1)^2 + 2 channels".into(),
                    );
                }
                if channels > 227 {
                    return Err("too many channels".into());
                }
            }
            vorbis = false;
        }
        channel_map = &ed[21..21 + channels];
    } else {
        return Err(format!("mapping type {map_type} is not supported"));
    }

    let reorder = |i: usize| -> usize {
        if vorbis {
            usize::from(VORBIS_CHANNEL_LAYOUT_OFFSETS[channels - 1][i])
        } else {
            i
        }
    };
    let mut maps = vec![ChannelMap::default(); channels];
    for i in 0..channels {
        let idx = usize::from(channel_map[reorder(i)]);
        let map = &mut maps[i];
        if idx == 255 {
            map.silence = true;
            continue;
        } else if idx >= streams + stereo_streams {
            return Err(format!("invalid channel map for output channel {i}: {idx}"));
        }
        map.copy = (0..i).find(|&j| usize::from(channel_map[reorder(j)]) == idx);
        if idx < 2 * stereo_streams {
            map.stream_idx = idx / 2;
            map.channel_idx = idx & 1;
        } else {
            map.stream_idx = idx - stereo_streams;
            map.channel_idx = 0;
        }
    }
    Ok(ParseContext {
        nb_streams: streams,
        nb_stereo_streams: stereo_streams,
        gain_i,
        channel_maps: maps,
        pre_skip,
    })
}

/// `ff_sqrt` for the ambisonic order.
fn ff_isqrt(v: usize) -> usize {
    super::rc::ff_sqrt(v as u32) as usize
}
