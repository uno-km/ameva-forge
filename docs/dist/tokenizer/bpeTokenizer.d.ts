/**
 * 파일 생성일: 2026-09-04
 * AMEVA-Forge Release 3.0: SCRUM-341 Universal Byte-Level BPE & SentencePiece Tokenizer
 *
 * WHAT: GGUF 메타데이터(tokenizer.ggml.tokens, merges) 또는 HuggingFace tokenizer.json으로부터
 *      어휘 사전(Vocabulary)과 BPE 규칙을 추출하여, 자연어 텍스트와 정수 토큰 ID 시퀀스 간의
 *      양방향 100% 무손실 인코딩/디코딩을 수행하는 범용 토크나이저입니다.
 * WHY: 침묵 토큰 누락이나 가짜 임베딩을 원천 차단하고, LLaMA-3, SmolLM, Gemma, Qwen 등
 *      다양한 오픈소스 LLM 가중치를 브라우저에서 플러그 앤 플레이로 즉시 구동하기 위함입니다.
 * HOW: UTF-8 바이트 분해 -> Regex 분할 -> BPE 병합 우선순위 룩업 -> 토큰 ID 생성 및 가역 복원.
 */
export declare enum TokenizerErrorCode {
    TOKENIZER_EMPTY_VOCAB = "TOKENIZER_EMPTY_VOCAB",
    TOKENIZER_NOT_INITIALIZED = "TOKENIZER_NOT_INITIALIZED",
    TOKENIZER_INVALID_TOKEN_ID = "TOKENIZER_INVALID_TOKEN_ID"
}
export declare class TokenizerError extends Error {
    readonly code: TokenizerErrorCode;
    constructor(code: TokenizerErrorCode, message: string);
}
export interface TokenizerConfig {
    vocab: string[];
    scores?: number[];
    merges?: string[];
    bosTokenId?: number;
    eosTokenId?: number;
    padTokenId?: number;
    unkTokenId?: number;
    chatTemplate?: string;
}
export declare class BPETokenizer {
    private vocabToId;
    private idToVocab;
    private bpeRanks;
    bosTokenId: number;
    eosTokenId: number;
    padTokenId: number;
    unkTokenId: number;
    isInitialized: boolean;
    constructor(config?: TokenizerConfig);
    private hasSentencePieceMarker;
    initFromConfig(config: TokenizerConfig): void;
    /**
     * GGUF 메타데이터 레코드로부터 토크나이저 어휘 사전 및 병합 규칙을 자동 추출합니다.
     */
    static fromGGUFMetadata(metadata: Record<string, any>): BPETokenizer;
    /**
     * 자연어 문자열을 토큰 ID 배열로 인코딩합니다.
     */
    encode(text: string, addBos?: boolean): number[];
    /**
     * 토큰 ID 배열을 디코딩하여 원본 자연어 문자열로 복원합니다.
     */
    decode(tokenIds: number[], skipSpecialTokens?: boolean): string;
    get vocabSize(): number;
}
