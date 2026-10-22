"""Adapters: forecasting libraries' models behind the `TimeSeriesModel` protocol.

Each adapter wraps one library's models, registers them under the library's namespace (such as
`statsforecast.AutoETS`), and imports the library only when a model is first fitted, so naming
a model costs nothing until it runs.
"""
