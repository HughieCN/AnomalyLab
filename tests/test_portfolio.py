import pandas as pd

from anomalylab.empirical.portfolio import PortfolioAnalysis
from anomalylab.structure import PanelData


def _portfolio_analysis() -> PortfolioAnalysis:
    df = pd.DataFrame(
        {
            "permno": [1, 2, 3, 4, 1, 2, 3, 4],
            "date": [
                "2020-01-31",
                "2020-01-31",
                "2020-01-31",
                "2020-01-31",
                "2020-02-29",
                "2020-02-29",
                "2020-02-29",
                "2020-02-29",
            ],
            "ret": [1.0, 3.0, 2.0, 6.0, 2.0, 6.0, 4.0, 12.0],
            "weight": [1.0] * 8,
            "signal": [10.0, 20.0, 10.0, 20.0, 10.0, 20.0, 10.0, 20.0],
            "signal_g2": [1, 2, 1, 2, 1, 2, 1, 2],
            "size": [100.0, 100.0, 200.0, 200.0, 100.0, 100.0, 200.0, 200.0],
            "size_g2": [1, 1, 2, 2, 1, 1, 2, 2],
        }
    )
    panel = PanelData(df=df, ret="ret")
    return PortfolioAnalysis(panel, endog="ret", weight="weight")


def test_univariate_analysis_can_return_low_minus_high_diff():
    portfolio = _portfolio_analysis()

    ew_ret, vw_ret = portfolio.univariate_analysis(
        "signal",
        2,
        already_grouped=True,
        factor_return=True,
        diff_direction="low-high",
    )

    assert ew_ret.loc[(pd.Period("2020-01", freq="M"), "Diff")] == -3.0
    assert vw_ret.loc[(pd.Period("2020-02", freq="M"), "Diff")] == -6.0


def test_univariate_analysis_defaults_to_high_minus_low_diff():
    portfolio = _portfolio_analysis()

    ew_ret, vw_ret = portfolio.univariate_analysis(
        "signal",
        2,
        already_grouped=True,
        factor_return=True,
    )

    assert ew_ret.loc[(pd.Period("2020-01", freq="M"), "Diff")] == 3.0
    assert vw_ret.loc[(pd.Period("2020-02", freq="M"), "Diff")] == 6.0


def test_bivariate_analysis_can_return_low_minus_high_diff():
    portfolio = _portfolio_analysis()

    ew_ret, vw_ret = portfolio.bivariate_analysis(
        "size",
        "signal",
        2,
        2,
        already_grouped=True,
        factor_return=True,
        diff_direction="low-high",
    )

    assert ew_ret.loc[(pd.Period("2020-01", freq="M"), 1), "Diff"] == -2.0
    assert ew_ret.loc[(pd.Period("2020-01", freq="M"), "Diff"), 1] == -1.0
    assert vw_ret.loc[(pd.Period("2020-02", freq="M"), "Diff"), "Diff"] == 4.0
