# Updating an integration test

If a change changes the output of an integration test beyond the usual tolerable limits (e.g. due to relaxation following a different path), the following can be done:

If the test uses `CliTest::check_file`, you can run with the `test-diff` feature to get a more complete overview of how the file changed.  For instance, if `tests/simple.rs` failed on a check of `raman.json`:

```
cargo test --features=test-diff -- --test simple
```

This should give a colorful diff, with highlighted character diffs giving an impression of how many significant figures are consistent between the files.

If everything looks reasonable, use RSP2_SAVETEMP to preserve the temp dir created by the integration test, and replace the test resource file.

```
RSP2_SAVETEMP=test-fail-tmp cargo test --features=test-diff -- --test simple
cp test-fail-tmp/rsp2.UbVPOAhWnzAN/out/raman.json tests/resources/simple-out/raman.json
```
