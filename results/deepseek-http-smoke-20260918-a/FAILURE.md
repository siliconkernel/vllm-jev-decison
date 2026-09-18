The first smoke attempt failed during HTTP client initialization, before model
inference: the inherited SOCKS proxy configuration required `socksio`, which was
not installed. Subsequent bridge clients set `trust_env=False` for direct access
to the configured upstream. No model result was produced in this run.
