/**
 * 파일 생성일: 2026-09-03
 * AMEVA-Forge Release 3.0: SCRUM-329 VAE Decoder Test Fixtures
 *
 * WHAT: 단위 테스트 전용 합성 가중치 생성 유틸리티입니다.
 * WHY: 운영 코드 및 배포 번들에 테스트 픽스처가 누출되는 것을 원천 차단하기 위해
 *      tests/fixtures/ 디렉토리에 엄격히 격리되었습니다.
 */

import { VAEDecoderWeights } from '../../src/diffusion/vaeDecoder';

export class VAEDecoderTestFixtures {
  public static createSyntheticWeights(inC: number = 4, midC: number = 32): VAEDecoderWeights {
    const createWeights = (ic: number, oc: number, k: number) => {
      const len = oc * ic * k * k;
      const w = new Float32Array(len);
      const std = Math.sqrt(2.0 / (ic * k * k));
      for (let i = 0; i < len; i++) {
        w[i] = ((i % 100) / 100.0 - 0.5) * std;
      }
      return w;
    };

    return {
      postQuantConvWeight: createWeights(inC, inC, 1),
      postQuantConvBias: new Float32Array(inC).fill(0.0),
      convInWeight: createWeights(inC, midC, 3),
      convInBias: new Float32Array(midC).fill(0.0),
      normOutGamma: new Float32Array(midC).fill(1.0),
      normOutBeta: new Float32Array(midC).fill(0.0),
      convOutWeight: createWeights(midC, 3, 3),
      convOutBias: new Float32Array(3).fill(0.0),
      upBlocks: [
        {
          upsampleConvWeight: createWeights(midC, midC, 3),
          upsampleConvBias: new Float32Array(midC).fill(0.0),
          normGamma: new Float32Array(midC).fill(1.0),
          normBeta: new Float32Array(midC).fill(0.0),
        },
        {
          upsampleConvWeight: createWeights(midC, midC, 3),
          upsampleConvBias: new Float32Array(midC).fill(0.0),
          normGamma: new Float32Array(midC).fill(1.0),
          normBeta: new Float32Array(midC).fill(0.0),
        },
        {
          upsampleConvWeight: createWeights(midC, midC, 3),
          upsampleConvBias: new Float32Array(midC).fill(0.0),
          normGamma: new Float32Array(midC).fill(1.0),
          normBeta: new Float32Array(midC).fill(0.0),
        },
      ],
    };
  }
}
