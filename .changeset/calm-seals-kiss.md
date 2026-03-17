---
"@martian-engineering/lossless-claw": patch
---

Declare explicit OpenClaw tool names for the LCM factory-registered tools so
plugin metadata and tool listings stay populated in hosts that require
`registerTool(..., { name })` hints for factory registrations.
