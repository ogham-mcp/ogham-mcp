# Database Platform Comparison: Bring Your Own Database

## Extension Support by Platform

| Platform | pgvector | pg_graphql | Self-managed | Notes |
|---|---|---|---|---|
| **Supabase (hosted)** | Yes | Yes | No | Core part of the stack |
| **Self-hosted (Docker)** | Yes | Yes | Yes | Use `supabase/postgres` or build from source |
| **Neon** | Yes | Yes | No | Now supported on paid/preview plans |
| **AWS Aurora / RDS** | Yes | No | No | Still lacks pg_graphql; use AWS AppSync for GraphQL |
| **Google Cloud SQL** | Yes | No | No | pgvector supported; pg_graphql not in allowed list |
| **Azure Flexible Server** | Yes | No | No | pgvector supported; no pg_graphql |
| **Any VPS (Hetzner, etc.)** | Yes | Yes | Yes | Total control; recommended for "Bring Your Own DB" |

## Key Takeaways

- **pgvector** is widely supported across all major managed Postgres providers.
- **pg_graphql** is a Supabase-built extension and is only available on Supabase hosted or self-hosted environments where you control the Postgres installation.
- If you need both extensions, your options are **Supabase hosted** or **self-hosted** (any server with Docker).
- If you only need vector search, any managed Postgres provider will work.

## Self-Hosted Options

Self-hosting gives full control over extensions. Two approaches:

1. **Docker (recommended)** — Use the `supabase/postgres` image which bundles pgvector, pg_graphql, and 30+ other extensions. Works on any host: Proxmox LXC, Hetzner VPS, AWS EC2, bare metal, etc.

2. **Native install** — Install Postgres directly and add extensions manually. pgvector is available as an apt package (`postgresql-17-pgvector`). pg_graphql requires building from source via Rust/pgrx, which is more complex.
