# Home lab build log

## ZFS pool

The tank pool uses two mirrored 4 TB drives. Scrubs run on the first Sunday of every month and snapshots are taken hourly for the code dataset. Compression is lz4 because the CPU cost is negligible and the space savings on text are large. Iteration note 1: measured again after changes and recorded the result here for later comparison.

The tank pool uses two mirrored 4 TB drives. Scrubs run on the first Sunday of every month and snapshots are taken hourly for the code dataset. Compression is lz4 because the CPU cost is negligible and the space savings on text are large. Iteration note 2: measured again after changes and recorded the result here for later comparison.

The tank pool uses two mirrored 4 TB drives. Scrubs run on the first Sunday of every month and snapshots are taken hourly for the code dataset. Compression is lz4 because the CPU cost is negligible and the space savings on text are large. Iteration note 3: measured again after changes and recorded the result here for later comparison.

The tank pool uses two mirrored 4 TB drives. Scrubs run on the first Sunday of every month and snapshots are taken hourly for the code dataset. Compression is lz4 because the CPU cost is negligible and the space savings on text are large. Iteration note 4: measured again after changes and recorded the result here for later comparison.

The tank pool uses two mirrored 4 TB drives. Scrubs run on the first Sunday of every month and snapshots are taken hourly for the code dataset. Compression is lz4 because the CPU cost is negligible and the space savings on text are large. Iteration note 5: measured again after changes and recorded the result here for later comparison.

## Networking

The server sits on the wired LAN at a fixed address. Remote access goes through Tailscale only; no ports are forwarded on the router. The firewall denies the Ollama port from everywhere except the Docker bridge networks. Iteration note 1: measured again after changes and recorded the result here for later comparison.

The server sits on the wired LAN at a fixed address. Remote access goes through Tailscale only; no ports are forwarded on the router. The firewall denies the Ollama port from everywhere except the Docker bridge networks. Iteration note 2: measured again after changes and recorded the result here for later comparison.

The server sits on the wired LAN at a fixed address. Remote access goes through Tailscale only; no ports are forwarded on the router. The firewall denies the Ollama port from everywhere except the Docker bridge networks. Iteration note 3: measured again after changes and recorded the result here for later comparison.

The server sits on the wired LAN at a fixed address. Remote access goes through Tailscale only; no ports are forwarded on the router. The firewall denies the Ollama port from everywhere except the Docker bridge networks. Iteration note 4: measured again after changes and recorded the result here for later comparison.

The server sits on the wired LAN at a fixed address. Remote access goes through Tailscale only; no ports are forwarded on the router. The firewall denies the Ollama port from everywhere except the Docker bridge networks. Iteration note 5: measured again after changes and recorded the result here for later comparison.

## Backups

Nightly backups copy the code and notes datasets to an external drive with restic. Once a month the external drive is rotated with a second one kept at the office, so a fire or theft never takes out every copy. Iteration note 1: measured again after changes and recorded the result here for later comparison.

Nightly backups copy the code and notes datasets to an external drive with restic. Once a month the external drive is rotated with a second one kept at the office, so a fire or theft never takes out every copy. Iteration note 2: measured again after changes and recorded the result here for later comparison.

Nightly backups copy the code and notes datasets to an external drive with restic. Once a month the external drive is rotated with a second one kept at the office, so a fire or theft never takes out every copy. Iteration note 3: measured again after changes and recorded the result here for later comparison.

Nightly backups copy the code and notes datasets to an external drive with restic. Once a month the external drive is rotated with a second one kept at the office, so a fire or theft never takes out every copy. Iteration note 4: measured again after changes and recorded the result here for later comparison.

Nightly backups copy the code and notes datasets to an external drive with restic. Once a month the external drive is rotated with a second one kept at the office, so a fire or theft never takes out every copy. Iteration note 5: measured again after changes and recorded the result here for later comparison.

## GPU

The graphics card has 12 GB of memory, enough for an 8B model at 4-bit quantization plus the embedding model at the same time. Larger models spill into system memory and become too slow for interactive use. Iteration note 1: measured again after changes and recorded the result here for later comparison.

The graphics card has 12 GB of memory, enough for an 8B model at 4-bit quantization plus the embedding model at the same time. Larger models spill into system memory and become too slow for interactive use. Iteration note 2: measured again after changes and recorded the result here for later comparison.

The graphics card has 12 GB of memory, enough for an 8B model at 4-bit quantization plus the embedding model at the same time. Larger models spill into system memory and become too slow for interactive use. Iteration note 3: measured again after changes and recorded the result here for later comparison.

The graphics card has 12 GB of memory, enough for an 8B model at 4-bit quantization plus the embedding model at the same time. Larger models spill into system memory and become too slow for interactive use. Iteration note 4: measured again after changes and recorded the result here for later comparison.

The graphics card has 12 GB of memory, enough for an 8B model at 4-bit quantization plus the embedding model at the same time. Larger models spill into system memory and become too slow for interactive use. Iteration note 5: measured again after changes and recorded the result here for later comparison.

## Commands

```bash
# check pool health
zpool status tank
```
