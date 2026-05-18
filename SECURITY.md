# Security Policy

## Reporting a Vulnerability

If you believe you've found a security vulnerability in `blind-ml` or in the demos it ships, please report it privately rather than opening a public issue.

**Email:** security@blindinsight.com

Please include:
- A description of the vulnerability and its potential impact
- Steps to reproduce (a minimal proof-of-concept is ideal)
- The version / commit hash you're testing against
- Any suggested mitigation

We aim to:
- Acknowledge your report within **2 business days**
- Provide an initial assessment within **5 business days**
- Coordinate a fix and disclosure timeline with you

## Disclosure

We follow a **coordinated disclosure** model. After we confirm a vulnerability and prepare a fix, we'll publish a security advisory (and a CVE where applicable) and credit you in the release notes — unless you prefer to remain anonymous.

Please give us a reasonable embargo window (typically 90 days from initial report, or sooner if a fix is available and shipped) before public disclosure.

## Scope

This policy covers:
- Code in this repository (notebooks, the `blind_ml` package, scripts)
- The demo data generators (synthetic data only; upload batches live in [demo-datasets](https://github.com/blind-insight/demo-datasets))

**Out of scope** (report to the respective project instead):
- Vulnerabilities in the Blind Insight platform itself (CLI, proxy, server) — see [docs.blindinsight.io](https://docs.blindinsight.io) for the platform's disclosure address
- Third-party dependencies — please report upstream

## Safe Harbor

We will not pursue legal action against researchers who:
- Make a good-faith effort to follow this policy
- Avoid privacy violations, data destruction, and service degradation
- Only interact with accounts they own or have explicit permission to test
- Give us reasonable time to fix the issue before public disclosure
