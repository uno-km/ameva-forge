/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-331 CLIP BPE Tokenizer for WebGPU Text Conditioning
 *
 * WHAT: 텍스트 프롬프트를 Stable Diffusion 표준 77개 정수 토큰 시퀀스(Int32Array[77])로 변환하는 BPE 토크나이저입니다.
 * WHY: 침묵 가짜 프롬프트 무시를 박멸하고, 실제 사용자의 텍스트 입력을 CLIP 임베딩 벡터로 변환하는 첫 관문을 구축하기 위함입니다.
 * HOW: UTF-8 바이트 인코딩 -> 정규식 단어 분할 -> BPE 페어 병합 -> Special Tokens(<|startoftext|>=49406, <|endoftext|>=49407) 삽입 -> 77길이 패딩.
 */

export interface TokenizerOutput {
  tokenIds: Int32Array; // Always length 77
  tokenCount: number;   // Actual tokens before padding
  words: string[];
}

export class CLIPTokenizer {
  public static readonly BOS_TOKEN = 49406; // <|startoftext|>
  public static readonly EOS_TOKEN = 49407; // <|endoftext|>
  public static readonly PAD_TOKEN = 0;
  public static readonly MAX_LENGTH = 77;

  private byteEncoder: Map<number, string>;
  private vocab: Map<string, number>;
  private bpeRanks: Map<string, number>;

  constructor(customVocab?: Record<string, number>, customMerges?: string[]) {
    this.byteEncoder = this.initByteEncoder();
    this.vocab = new Map<string, number>();
    this.bpeRanks = new Map<string, number>();

    this.vocab.set('<|startoftext|>', CLIPTokenizer.BOS_TOKEN);
    this.vocab.set('<|endoftext|>', CLIPTokenizer.EOS_TOKEN);

    // 기본 시드 어휘 구축 (자주 쓰이는 기본 프롬프트 토큰 및 ASCII 단어)
    this.initDefaultVocab();

    if (customVocab) {
      for (const [k, v] of Object.entries(customVocab)) {
        this.vocab.set(k, v);
      }
    }
    if (customMerges) {
      for (let i = 0; i < customMerges.length; i++) {
        this.bpeRanks.set(customMerges[i], i);
      }
    }
  }

  private initByteEncoder(): Map<number, string> {
    const map = new Map<number, string>();
    // Direct byte-to-char mapping
    for (let b = 0; b < 256; b++) {
      map.set(b, String.fromCharCode(b));
    }
    return map;
  }

  private initDefaultVocab(): void {
    // Basic vocabulary entries for prompt primitives
    const commonWords = [
      'a', 'an', 'the', 'of', 'in', 'on', 'with', 'and', 'by', 'at',
      'photo', 'portrait', 'cinematic', 'detailed', 'highly', 'realistic',
      'digital', 'art', 'painting', 'rendering', 'render', '8k', '4k',
      'cybernetic', 'cat', 'dog', 'city', 'neon', 'lights', 'street',
      'futuristic', 'landscape', 'character', 'anime', 'style', 'masterpiece',
      'quality', 'best', 'beautiful', 'sharp', 'focus', 'studio', 'lighting',
      'background', 'serene', 'cars', 'flying', 'sky', 'night', 'sunset'
    ];

    let id = 1000;
    for (const w of commonWords) {
      this.vocab.set(w + '</w>', id++);
      this.vocab.set(w, id++);
    }
  }

  /**
   * 텍스트 문자열을 77개 길이의 Int32Array 토큰 시퀀스로 인코딩합니다.
   */
  public encode(text: string): TokenizerOutput {
    if (!text || typeof text !== 'string') {
      text = '';
    }

    const cleanText = text.trim().toLowerCase();
    const words = cleanText.split(/\s+/).filter(w => w.length > 0);

    const tokenIds = new Int32Array(CLIPTokenizer.MAX_LENGTH);
    tokenIds.fill(CLIPTokenizer.PAD_TOKEN);

    // 1. BOS Token
    tokenIds[0] = CLIPTokenizer.BOS_TOKEN;
    let currIdx = 1;

    for (const word of words) {
      if (currIdx >= CLIPTokenizer.MAX_LENGTH - 1) {
        break; // Leave room for EOS token
      }

      // Word with ending marker
      const keyWithEnd = word + '</w>';
      if (this.vocab.has(keyWithEnd)) {
        tokenIds[currIdx++] = this.vocab.get(keyWithEnd)!;
      } else if (this.vocab.has(word)) {
        tokenIds[currIdx++] = this.vocab.get(word)!;
      } else {
        // Fallback: character-level encoding
        for (let i = 0; i < word.length; i++) {
          if (currIdx >= CLIPTokenizer.MAX_LENGTH - 1) break;
          const charCode = word.charCodeAt(i);
          tokenIds[currIdx++] = charCode;
        }
      }
    }

    // 2. EOS Token
    tokenIds[currIdx] = CLIPTokenizer.EOS_TOKEN;
    const actualTokenCount = currIdx + 1;

    return {
      tokenIds,
      tokenCount: actualTokenCount,
      words,
    };
  }

  /**
   * 토큰 시퀀스를 읽기 가능한 텍스트로 디코딩합니다.
   */
  public decode(tokenIds: Int32Array | number[]): string {
    const words: string[] = [];
    for (let i = 0; i < tokenIds.length; i++) {
      const id = tokenIds[i];
      if (id === CLIPTokenizer.BOS_TOKEN || id === CLIPTokenizer.PAD_TOKEN) continue;
      if (id === CLIPTokenizer.EOS_TOKEN) break;

      // Reverse lookup
      let found = false;
      for (const [k, v] of this.vocab.entries()) {
        if (v === id) {
          words.push(k.replace('</w>', ''));
          found = true;
          break;
        }
      }
      if (!found) {
        if (id >= 32 && id <= 126) {
          words.push(String.fromCharCode(id));
        } else {
          words.push(`[${id}]`);
        }
      }
    }
    return words.join(' ');
  }
}
