"""Alpha101 运算子测试（小数组，对照手算 / pandas 结果）。"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "research_loop"))
import alpha101 as A          # noqa: E402

X = np.array([[1.0, 5.0], [3.0, 4.0], [2.0, 3.0], [5.0, 2.0], [4.0, 1.0]])


def test_delay_delta_floor_window():
    assert np.isnan(A.delay(X, 2.7)[1, 0]) and A.delay(X, 2.7)[2, 0] == 1.0      # 2.7 → 2
    assert A.delta(X, 1)[1, 0] == 2.0


def test_ts_rank_and_argmax():
    r = A.ts_rank(X, 3)
    assert r[2, 0] == pytest.approx(2 / 3)        # 窗口 [1,3,2]，今天 2 排第 2
    assert r[3, 0] == pytest.approx(1.0)          # [3,2,5]，今天最大
    assert A.ts_argmax(X, 3)[2, 0] == 2.0         # 最大值 3 出现在窗口第 2 天
    assert A.ts_argmin(X, 3)[4, 1] == 3.0         # 第二列递减，今天最小


def test_decay_linear_weights():
    v = A.decay_linear(X, 3)[2, 0]
    assert v == pytest.approx((3 * 2 + 2 * 3 + 1 * 1) / 6)


def test_correlation_matches_pandas():
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=(30, 3)), rng.normal(size=(30, 3))
    got = A.correlation(a, b, 10)
    ref = pd.DataFrame(a).rolling(10).corr(pd.DataFrame(b)).to_numpy()
    assert np.allclose(got[9:], ref[9:], atol=1e-9)


def test_scale_and_rank():
    s = A.scale(np.array([[1.0, -3.0]]))
    assert np.abs(s).sum() == pytest.approx(1.0)
    assert A.rank(np.array([[3.0, 1.0, 2.0]])).tolist() == [[1.0, 1 / 3, 2 / 3]]
