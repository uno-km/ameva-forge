#!/usr/bin/env python3
"""
Updates docs/i18n-translations.js with Release 2.0 LLM & GEMM translation keys across all 6 languages.
"""

import json
import re

r2_translations = {
    "en": {
        "tiledMatmulTitle": "16x16 Shared Memory Tiled MatMul",
        "tiledMatmulDesc": "Accelerates General Matrix Multiplication (GEMM) by 3.5x-5x using workgroup shared memory tiles.",
        "flashAttentionTitle": "FlashAttention-2 Fused 1-Pass Kernel",
        "flashAttentionDesc": "Eliminates O(N^2) VRAM allocation via in-register online softmax and causal masking.",
        "ropeTitle": "Rotary Position Embedding (RoPE)",
        "ropeDesc": "Injects positional encoding into query/key vectors via complex plane 2D rotation.",
        "rmsnormTitle": "Root Mean Square Normalization (RMSNorm)",
        "rmsnormDesc": "Fast 256-thread tree reduction normalization with learnable gamma scaling.",
        "swigluTitle": "SwiGLU Fused Activation",
        "swigluDesc": "Fused Swish(x) * y gating for state-of-the-art LLM Feed-Forward Networks.",
        "quantizationTitle": "INT4 / INT8 Quantized Weights",
        "quantizationDesc": "Real-time on-the-fly dequantization enabling 7B LLM inference within 4GB VRAM."
    },
    "ko": {
        "tiledMatmulTitle": "16x16 공유 메모리 타일드 행렬곱 (Tiled MatMul)",
        "tiledMatmulDesc": "워크그룹 공유 메모리 타일을 활용하여 GEMM 연산 속도를 3.5배~5배 가속합니다.",
        "flashAttentionTitle": "FlashAttention-2 융합 1-Pass 커널",
        "flashAttentionDesc": "레지스터 레벨 온라인 소프트맥스와 인커널 인과적 마스킹으로 O(N^2) VRAM 할당을 제거합니다.",
        "ropeTitle": "회전 위치 임베딩 (RoPE)",
        "ropeDesc": "2D 복소 평면 회전을 통해 Query와 Key 벡터에 토큰 위치 정보를 인플레이스 주입합니다.",
        "rmsnormTitle": "RMS 정규화 (RMSNorm)",
        "rmsnormDesc": "256 스레드 트리 리덕션과 학습 가능한 감마 스케일링으로 고속 정규화를 수행합니다.",
        "swigluTitle": "SwiGLU 융합 활성화 함수",
        "swigluDesc": "Swish(x) * y 게이팅을 단일 패스로 처리하여 최신 LLM FFN을 가속합니다.",
        "quantizationTitle": "INT4 / INT8 가중치 역양자화",
        "quantizationDesc": "실시간 온더플라이 역양자화로 7B 대규모 언어모델을 4GB 미만 VRAM에서 구동합니다."
    },
    "zh": {
        "tiledMatmulTitle": "16x16 共享内存分块矩阵乘法 (Tiled MatMul)",
        "tiledMatmulDesc": "利用工作组共享内存分块将通用矩阵乘法 (GEMM) 速度提升 3.5 至 5 倍。",
        "flashAttentionTitle": "FlashAttention-2 融合单遍算子",
        "flashAttentionDesc": "通过寄存器在线 Softmax 和因果掩码彻底消除 O(N^2) 显存占用。",
        "ropeTitle": "旋转位置编码 (RoPE)",
        "ropeDesc": "通过复平面 2D 旋转将位置信息原位注入 Query 和 Key 向量。",
        "rmsnormTitle": "均方根归一化 (RMSNorm)",
        "rmsnormDesc": "基于 256 线程树状约简的高速归一化，支持可学习 Gamma 缩放。",
        "swigluTitle": "SwiGLU 融合激活函数",
        "swigluDesc": "单遍执行 Swish(x) * y 门控计算，加速前沿大模型 FFN 网络。",
        "quantizationTitle": "INT4 / INT8 量化权重解包",
        "quantizationDesc": "实时片上解量化，使 7B 大语言模型能够在 4GB 显存内流畅运行。"
    },
    "ja": {
        "tiledMatmulTitle": "16x16 共有メモリ タイルド行列積 (Tiled MatMul)",
        "tiledMatmulDesc": "ワークグループ共有メモリを活用して GEMM 演算速度を 3.5〜5 倍に高速化します。",
        "flashAttentionTitle": "FlashAttention-2 融合 1-Pass カーネル",
        "flashAttentionDesc": "レジスタ内オンライン Softmax と因果マスクにより O(N^2) VRAM 割り当てを完全に排除します。",
        "ropeTitle": "回転位置埋め込み (RoPE)",
        "ropeDesc": "複素平面 2D 回転により Query と Key ベクトルに位置情報をインプレースで注入します。",
        "rmsnormTitle": "二乗平均平方根正規化 (RMSNorm)",
        "rmsnormDesc": "256 スレッド ツリーリダクションによる高速正規化と学習可能 Gamma スケーリングを提供します。",
        "swigluTitle": "SwiGLU 融合活性化関数",
        "swigluDesc": "Swish(x) * y ゲーティングを単一パスで処理し、最新 LLM FFN を高速化します。",
        "quantizationTitle": "INT4 / INT8 量子化重み逆量子化",
        "quantizationDesc": "リアルタイムオンザフライ逆量子化により、4GB 未満の VRAM で 7B LLM を実行可能です。"
    },
    "hi": {
        "tiledMatmulTitle": "16x16 शेयर्ड मेमोरी टाइल्ड मैट्रिक्स मल्टिप्लिकेशन",
        "tiledMatmulDesc": "वर्कग्रुप शेयर्ड मेमोरी टाइल्स का उपयोग करके GEMM ऑपरेशंस को 3.5x-5x तेज करता है।",
        "flashAttentionTitle": "FlashAttention-2 फ्यूज्ड 1-पास कर्नेल",
        "flashAttentionDesc": "रजिस्टर-लेवल ऑनलाइन सॉफ्टमैक्स और कॉज़ल मास्किंग के माध्यम से O(N^2) VRAM उपयोग को समाप्त करता है।",
        "ropeTitle": "रोटरी पोजीशन एम्बेडिंग (RoPE)",
        "ropeDesc": "जटिल 2D रोटेशन के माध्यम से क्वेरी और की वैक्टर में स्थिति जानकारी इंजेक्ट करता है।",
        "rmsnormTitle": "रूट मीन स्क्वायर नॉर्मलाइज़ेशन (RMSNorm)",
        "rmsnormDesc": "256-थ्रेड ट्री रिडक्शन के साथ तेज़ नॉर्मलाइज़ेशन और स्केलिंग प्रदान करता है।",
        "swigluTitle": "SwiGLU फ्यूज्ड एक्टिवेशन",
        "swigluDesc": "आधुनिक LLM FFN ब्लॉक के लिए Swish(x) * y गेटिंग को सिंगल पास में निष्पादित करता है।",
        "quantizationTitle": "INT4 / INT8 क्वांटाइज़्ड वेट्स",
        "quantizationDesc": "रीयल-टाइम डीक्वांटाइज़ेशन जो 4GB VRAM के भीतर 7B LLM चलाने में सक्षम बनाता है।"
    },
    "es": {
        "tiledMatmulTitle": "Multiplicación de Matrices en Mosaico con Memoria Compartida 16x16",
        "tiledMatmulDesc": "Acelera las operaciones GEMM de 3.5x a 5x mediante mosaicos de memoria compartida.",
        "flashAttentionTitle": "Kernel Fusionado FlashAttention-2 de 1 Pase",
        "flashAttentionDesc": "Elimina la asignación O(N^2) de VRAM mediante softmax en línea en registros y enmascaramiento causal.",
        "ropeTitle": "Incrustación de Posición Rotatoria (RoPE)",
        "ropeDesc": "Inyecta información posicional en vectores Query y Key mediante rotación 2D en el plano complejo.",
        "rmsnormTitle": "Normalización de Raíz Cuadrada Media (RMSNorm)",
        "rmsnormDesc": "Normalización rápida mediante reducción en árbol de 256 hilos con escala Gamma aprendible.",
        "swigluTitle": "Activación Fusionada SwiGLU",
        "swigluDesc": "Procesa la compuerta Swish(x) * y en un solo pase para redes FFN de modelos LLM avanzados.",
        "quantizationTitle": "Pesos Cuantizados INT4 / INT8",
        "quantizationDesc": "Descuantización en tiempo real que permite inferir modelos LLM 7B con menos de 4GB de VRAM."
    }
}

def update_file():
    with open("docs/i18n-translations.js", "r", encoding="utf-8") as f:
        content = f.read()

    # In each language block, find playgrounds: { ... } and add r2: { ... },
    for lang, data in r2_translations.items():
        json_r2 = json.dumps(data, ensure_ascii=False, indent=16)
        r2_block = f'r2: {json_r2},\n            playgrounds:'
        
        # Regex to match playgrounds inside the specific language section
        pattern = rf'({lang}\s*:\s*\{{[\s\S]*?)(playgrounds\s*:)'
        match = re.search(pattern, content)
        if match:
            # Check if r2 already exists
            if f'{lang}:' in content and 'r2:' in match.group(1):
                continue
            content = content[:match.start(2)] + f'r2: {json_r2},\n            ' + content[match.start(2):]

    with open("docs/i18n-translations.js", "w", encoding="utf-8") as f:
        f.write(content)
    print("[+] Successfully updated docs/i18n-translations.js with Release 2.0 keys!")

if __name__ == "__main__":
    update_file()
