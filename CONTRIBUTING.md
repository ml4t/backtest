# Contributing to ml4t-backtest

Read the [ML4T contribution policy](https://github.com/ml4t/ecosystem/blob/main/CONTRIBUTING.md)
before opening an issue or pull request. This guide identifies the Backtest-specific entry points.

Search [existing issues](https://github.com/ml4t/backtest/issues) first. Report one problem per
issue. For a bug, include the affected package version, Python and operating-system versions,
expected and observed behavior, and a complete example that reproduces the result. Report suspected
vulnerabilities through the [private security process](SECURITY.md).

Keep each pull request focused on one issue and include a closing reference such as `Closes #123`.
Automated dependency updates and administrative metadata changes do not need a separate issue. For
a bug fix, run a regression through the affected public API or user workflow against both the base
revision and the fix. Record both revisions, commands, and results in the pull request. The test
must fail because of the defect on the base and pass with the fix. Documentation changes need
documentation checks, not an artificial regression test.

Run the relevant commands in the [Development section](README.md#development) and the
[repository quality commands](AGENTS.md#quality-commands), then report the results. CI also checks
compatibility, security, parity, and package artifacts. State whether AI tools helped with the issue
or pull request and which parts they produced. Review and test that work yourself; be prepared to
explain it during review.
