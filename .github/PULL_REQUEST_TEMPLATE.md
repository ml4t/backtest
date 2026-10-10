## Owning issue

Closes #

Keep this PR focused on one issue. Automated dependency updates and administrative metadata
changes may state that exception here instead of linking an issue.

## Outcome

Describe the user-visible or standards outcome.

## Bug regression evidence

For a bug fix, exercise the affected public API or user workflow on both revisions. The test must
fail because of the defect on the base and pass with the fix. State "Not a bug fix" when this
section does not apply.

- Base revision, command, and observed failure:
- Fix revision, command, and passing result:

## Compatibility

- [ ] No public compatibility impact
- [ ] Compatibility impact is documented in the owning issue and migration guidance

## Verification

- [ ] Bug regression fails on the base and passes with the fix, or this is not a bug fix
- [ ] Ruff lint and format checks pass
- [ ] `ty` passes
- [ ] Test suite passes
- [ ] Package build passes when applicable
- [ ] Strict MkDocs build passes when documentation is affected
- [ ] Ecosystem qualification passes

List the commands actually run and their results. Explain any unchecked item.

## AI assistance

State whether AI tools helped produce the code, tests, documentation, or PR text, and identify the
parts they produced. Write "None" if none. Confirm that you reviewed and tested any AI-assisted work.

## Documentation and release

State the documentation, release notes, and patch-release work required after merge.
