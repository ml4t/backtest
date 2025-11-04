# ==================================== VBTPROXYZ ====================================
# Copyright (c) 2021-2025 Oleg Polakow. All rights reserved.
#
# This file is part of the proprietary VectorBT® PRO package and is licensed under
# the VectorBT® PRO License available at https://vectorbt.pro/terms/software-license/
#
# Unauthorized publishing, distribution, sublicensing, or sale of this software
# or its parts is strictly prohibited.
# ===================================================================================

"""Module providing configuration for optional dependencies for internal use."""

from vectorbtpro.utils.config import HybridConfig

__all__ = []

__pdoc__ = {}

opt_dep_config = HybridConfig(
    dict(
        yfinance=dict(version=">=0.2.56"),
        binance=dict(dist_name="python-binance", version=">=1.0.16"),
        ccxt=dict(version=">=1.89.14"),
        ta=dict(),
        pandas_ta=dict(),
        talib=dict(dist_name="TA-Lib"),
        bottleneck=dict(),
        numexpr=dict(),
        ray=dict(version=">=1.4.1"),
        dask=dict(),
        matplotlib=dict(version=">=3.2.0"),
        plotly=dict(version=">=5.0.0"),
        ipywidgets=dict(version=">=7.0.0"),
        kaleido=dict(),
        telegram=dict(dist_name="python-telegram-bot", version=">=13.4"),
        quantstats=dict(version=">=0.0.37"),
        dill=dict(),
        alpaca=dict(dist_name="alpaca-py", version=">=0.40.0"),
        polygon=dict(dist_name="polygon-api-client", version=">=1.0.0"),
        bs4=dict(dist_name="beautifulsoup4"),
        nasdaqdatalink=dict(dist_name="Nasdaq-Data-Link"),
        pypfopt=dict(dist_name="PyPortfolioOpt", version=">=1.5.1"),
        universal=dict(dist_name="universal-portfolios"),
        plotly_resampler=dict(dist_name="plotly-resampler"),
        technical=dict(),
        riskfolio=dict(dist_name="Riskfolio-Lib", version=">=3.3.0"),
        pathos=dict(),
        lz4=dict(),
        blosc=dict(),
        blosc2=dict(),
        tables=dict(),
        optuna=dict(),
        sqlalchemy=dict(dist_name="SQLAlchemy", version=">=2.0.0"),
        mpire=dict(),
        duckdb=dict(),
        duckdb_engine=dict(dist_name="duckdb-engine"),
        pyarrow=dict(),
        fastparquet=dict(),
        tabulate=dict(),
        alpha_vantage=dict(version=">=3.0.0"),
        databento=dict(),
        smartmoneyconcepts=dict(),
        findatapy=dict(),
        github=dict(dist_name="PyGithub", version=">=1.59.0"),
        jmespath=dict(),
        jsonpath_ng=dict(dist_name="jsonpath-ng"),
        fuzzysearch=dict(),
        rapidfuzz=dict(),
        nestedtext=dict(),
        yaml=dict(dist_name="PyYAML"),
        ruamel=dict(dist_name="ruamel.yaml"),
        toml=dict(),
        markdown=dict(),
        pygments=dict(),
        IPython=dict(dist_name="ipython"),
        pymdownx=dict(dist_name="pymdown-extensions"),
        openai=dict(),
        litellm=dict(),
        llama_index=dict(dist_name="llama-index"),
        tiktoken=dict(),
        lmdbm=dict(),
        bm25s=dict(),
        PyStemmer=dict(),
        pyperclip=dict(),
        platformdirs=dict(),
        mcp=dict(),
        jupyter_client=dict(),
    )
)
"""_"""

__pdoc__[
    "opt_dep_config"
] = f"""Configuration for optional dependencies used internally by vectorbtpro.

Contains package metadata including download links, version requirements, and distribution names where applicable.

```python
{opt_dep_config.prettify_doc()}
```
"""
