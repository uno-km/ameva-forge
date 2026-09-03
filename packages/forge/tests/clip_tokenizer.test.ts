/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-331 CLIP Tokenizer Unit Tests
 */

import { CLIPTokenizer } from '../src/diffusion/clipTokenizer';

describe('CLIPTokenizer Unit Tests (SCRUM-331)', () => {
  let tokenizer: CLIPTokenizer;

  beforeEach(() => {
    tokenizer = new CLIPTokenizer();
  });

  it('encodes prompt with BOS and EOS tokens and pads to 77 tokens', () => {
    const prompt = 'a cinematic portrait of a cybernetic cat';
    const out = tokenizer.encode(prompt);

    expect(out.tokenIds.length).toBe(77);
    expect(out.tokenIds[0]).toBe(CLIPTokenizer.BOS_TOKEN);
    expect(out.tokenIds[out.tokenCount - 1]).toBe(CLIPTokenizer.EOS_TOKEN);
    expect(out.tokenIds[76]).toBe(CLIPTokenizer.PAD_TOKEN);
  });

  it('safely handles empty string or whitespace prompt', () => {
    const out = tokenizer.encode('   ');
    expect(out.tokenIds.length).toBe(77);
    expect(out.tokenIds[0]).toBe(CLIPTokenizer.BOS_TOKEN);
    expect(out.tokenIds[1]).toBe(CLIPTokenizer.EOS_TOKEN);
  });

  it('truncates long prompt to strictly fit within 77 tokens', () => {
    const longPrompt = Array(120).fill('cybernetic').join(' ');
    const out = tokenizer.encode(longPrompt);

    expect(out.tokenIds.length).toBe(77);
    expect(out.tokenIds[0]).toBe(CLIPTokenizer.BOS_TOKEN);
    expect(out.tokenIds[76]).toBe(CLIPTokenizer.EOS_TOKEN);
  });

  it('decodes token sequence back to words', () => {
    const prompt = 'neon city lights';
    const out = tokenizer.encode(prompt);
    const decoded = tokenizer.decode(out.tokenIds);

    expect(decoded).toContain('neon');
    expect(decoded).toContain('city');
    expect(decoded).toContain('lights');
  });
});
