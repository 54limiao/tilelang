import argparse

import numpy as np

from examples.qwen3_int_only.utils.proto import attention_i8, linear_i16, linear_i8, quant_v_i8, rms_q15, rms_sq8, rope_sq8, silu_i16


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    rng = np.random.default_rng(args.seed)
    quant_v_i8.check(rng)
    rms_q15.check(rng)
    rms_sq8.check(rng)
    rope_sq8.check(rng)
    silu_i16.check(rng)
    linear_i8.check(rng)
    linear_i16.check(rng)
    attention_i8.check(rng)


if __name__ == "__main__":
    main()
