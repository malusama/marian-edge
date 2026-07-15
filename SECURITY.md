# Security policy

## Supported versions

Security fixes are applied to the latest tagged release and `main`. Pre-1.0
releases may contain breaking changes.

## Reporting a vulnerability

Please use the repository's private **Security > Report a vulnerability**
form:

<https://github.com/malusama/marian-edge/security/advisories/new>

Do not file a public issue for a suspected vulnerability. Include the affected
version, operating system or image digest, reproduction steps, and realistic
impact. We aim to acknowledge reports within seven days.

## Deployment boundary

The native server defaults to a loopback-only listener. The container listens
on `0.0.0.0:3000` inside its network namespace, while the supported Compose
file publishes that port only on host `127.0.0.1`. The service has no
authentication, authorization, TLS termination, tenant isolation, or abuse
controls. Do not bind or publish it to a public or untrusted network. Put an
authenticated reverse proxy and request limits in front of it if remote access
is required.

Text is processed locally and is not intentionally logged, but operators are
responsible for host, proxy, and container logging policies.
