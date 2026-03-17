---
"@martian-engineering/lossless-claw": patch
---

Improve LCM summarization model resolution so configured `summaryModel`
overrides, OpenClaw `agents.defaults.compaction.model`, and newer
`runtimeContext` inputs are honored more reliably while preserving
compatibility with older `legacyCompactionParams` integrations.
