// Minimal GGUF v3 reader for Turp (F32 / F16 tensors).
// Parses the same file llama.cpp produces; no external dependencies.

const textDecoder = new TextDecoder();
const textEncoder = new TextEncoder();

function halfToFloat(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * Math.pow(2, -14) * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * Math.pow(2, e - 15) * (1 + f / 1024);
}

function align(n, a) {
  return Math.ceil(n / a) * a;
}

export function parseGGUF(buffer) {
  const dv = new DataView(buffer);
  let o = 0;
  const u8 = () => dv.getUint8(o++);
  const i8 = () => dv.getInt8(o++);
  const u16 = () => { const v = dv.getUint16(o, true); o += 2; return v; };
  const i16 = () => { const v = dv.getInt16(o, true); o += 2; return v; };
  const u32 = () => { const v = dv.getUint32(o, true); o += 4; return v; };
  const i32 = () => { const v = dv.getInt32(o, true); o += 4; return v; };
  const f32 = () => { const v = dv.getFloat32(o, true); o += 4; return v; };
  const u64 = () => { const v = Number(dv.getBigUint64(o, true)); o += 8; return v; };
  const i64 = () => { const v = Number(dv.getBigInt64(o, true)); o += 8; return v; };
  const f64 = () => { const v = dv.getFloat64(o, true); o += 8; return v; };
  const str = () => {
    const n = u64();
    const s = textDecoder.decode(new Uint8Array(buffer, o, n));
    o += n;
    return s;
  };
  const value = (t) => {
    switch (t) {
      case 0: return u8();
      case 1: return i8();
      case 2: return u16();
      case 3: return i16();
      case 4: return u32();
      case 5: return i32();
      case 6: return f32();
      case 7: return u8() !== 0;
      case 8: return str();
      case 9: {
        const et = u32();
        const n = u64();
        const arr = new Array(n);
        for (let i = 0; i < n; i++) arr[i] = value(et);
        return arr;
      }
      case 10: return u64();
      case 11: return i64();
      case 12: return f64();
      default: throw new Error("gguf: unknown value type " + t);
    }
  };

  const magic = textDecoder.decode(new Uint8Array(buffer, 0, 4));
  if (magic !== "GGUF") throw new Error("not a GGUF file");
  o = 4;
  const version = u32();
  if (version !== 3) throw new Error("unsupported GGUF version " + version);
  const tensorCount = u64();
  const kvCount = u64();

  const metadata = {};
  for (let i = 0; i < kvCount; i++) {
    const key = str();
    const type = u32();
    metadata[key] = value(type);
  }

  const alignment = metadata["general.alignment"] || 32;
  const infos = [];
  for (let i = 0; i < tensorCount; i++) {
    const name = str();
    const nDims = u32();
    const dims = [];
    for (let d = 0; d < nDims; d++) dims.push(u64());
    const type = u32();
    const offset = u64();
    infos.push({ name, dims, type, offset });
  }

  const dataStart = align(o, alignment);
  const tensors = new Map();

  for (const info of infos) {
    const count = info.dims.reduce((a, b) => a * b, 1);
    const start = dataStart + info.offset;
    let data;
    if (info.type === 0) {
      data = new Float32Array(count);
      for (let i = 0; i < count; i++) data[i] = dv.getFloat32(start + i * 4, true);
    } else if (info.type === 1) {
      data = new Float32Array(count);
      for (let i = 0; i < count; i++) data[i] = halfToFloat(dv.getUint16(start + i * 2, true));
    } else {
      throw new Error("gguf: unsupported tensor type " + info.type + " for " + info.name);
    }
    tensors.set(info.name, { dims: info.dims, data });
  }

  return { version, metadata, tensors };
}

export function getTensor(gguf, name) {
  const t = gguf.tensors.get(name);
  if (!t) throw new Error("gguf: missing tensor " + name);
  return t;
}
