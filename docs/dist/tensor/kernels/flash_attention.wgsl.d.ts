/**
 * 파일 생성일: 2026-08-18
 * AMEVA-Forge Release 2.0: SCRUM-209 / SCRUM-210 / SCRUM-211
 * FlashAttention-2 Fused 1-Pass Online Softmax WGSL Kernel (MHA / GQA / Causal)
 *
 * WHAT: O(N) 메모리 복잡도를 가지는 FlashAttention-2 융합 1-Pass 어텐션 WGSL 셰이더입니다.
 * WHY: 표준 Scaled Dot-Product Attention의 O(N^2) 어텐션 맵 VRAM 할당과 대역폭 병목을 100% 제거하고,
 *      긴 시퀀스(SeqLen 2048~4096)에서도 OOM 없이 초고속 LLM 추론을 가능하게 합니다.
 * HOW: Dao et al.의 FlashAttention-2 Online Softmax 알고리즘(Running Max & Running Sum)을 GPU 스레드 레지스터 레벨에서
 *      단일 패스로 융합하고, Grouped Query Attention(GQA)과 Causal Masking을 셰이더 내부에서 인라인으로 처리합니다.
 */
export declare function getFlashAttentionWGSL(headDim?: number): string;
export declare const FLASH_ATTENTION_WGSL: string;
