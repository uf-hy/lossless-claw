import { randomUUID } from "node:crypto";
import { readFileSync } from "node:fs";
import { mkdir, writeFile } from "node:fs/promises";
import { homedir } from "node:os";
import { join } from "node:path";
import type {
  ContextEngine,
  ContextEngineInfo,
  AssembleResult,
  BootstrapResult,
  CompactResult,
  IngestBatchResult,
  IngestResult,
  SubagentEndReason,
  SubagentSpawnPreparation,
} from "openclaw/plugin-sdk";
import { ContextAssembler } from "./assembler.js";
import { CompactionEngine, type CompactionConfig } from "./compaction.js";
import type { LcmConfig } from "./db/config.js";
import { getLcmConnection, closeLcmConnection } from "./db/connection.js";
import { getLcmDbFeatures } from "./db/features.js";
import { runLcmMigrations } from "./db/migration.js";
import {
  createDelegatedExpansionGrant,
  removeDelegatedExpansionGrantForSession,
  revokeDelegatedExpansionGrantForSession,
} from "./expansion-auth.js";
import {
  extensionFromNameOrMime,
  formatFileReference,
  generateExplorationSummary,
  parseFileBlocks,
} from "./large-files.js";
import { RetrievalEngine } from "./retrieval.js";
import {
  ConversationStore,
  type CreateMessagePartInput,
  type MessagePartType,
} from "./store/conversation-store.js";
import { SummaryStore } from "./store/summary-store.js";
import { createLcmSummarizeFromLegacyParams } from "./summarize.js";
import type { LcmDependencies } from "./types.js";

type AgentMessage = Parameters<ContextEngine["ingest"]>[0]["message"];
type AssembleResultWithSystemPrompt = AssembleResult & { systemPromptAddition?: string };

// ── Helpers ──────────────────────────────────────────────────────────────────

/** Rough token estimate: ~4 chars per token. */
function estimateTokens(text: string): number {
  return Math.ceil(text.length / 4);
}

function toJson(value: unknown): string {
  const encoded = JSON.stringify(value);
  return typeof encoded === "string" ? encoded : "";
}

function safeString(value: unknown): string | undefined {
  return typeof value === "string" ? value : undefined;
}

function safeBoolean(value: unknown): boolean | undefined {
  return typeof value === "boolean" ? value : undefined;
}

function appendTextValue(value: unknown, out: string[]): void {
  if (typeof value === "string") {
    out.push(value);
    return;
  }
  if (Array.isArray(value)) {
    for (const entry of value) {
      appendTextValue(entry, out);
    }
    return;
  }
  if (!value || typeof value !== "object") {
    return;
  }

  const record = value as Record<string, unknown>;
  appendTextValue(record.text, out);
  appendTextValue(record.value, out);
}

function extractReasoningText(record: Record<string, unknown>): string | undefined {
  const chunks: string[] = [];
  appendTextValue(record.summary, chunks);
  if (chunks.length === 0) {
    return undefined;
  }

  const normalized = chunks
    .map((chunk) => chunk.trim())
    .filter((chunk, idx, arr) => chunk.length > 0 && arr.indexOf(chunk) === idx);
  return normalized.length > 0 ? normalized.join("\n") : undefined;
}

function normalizeUnknownBlock(value: unknown): {
  type: string;
  text?: string;
  metadata: Record<string, unknown>;
} {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return {
      type: "agent",
      metadata: { raw: value },
    };
  }

  const record = value as Record<string, unknown>;
  const rawType = safeString(record.type);
  return {
    type: rawType ?? "agent",
    text:
      safeString(record.text) ??
      safeString(record.thinking) ??
      ((rawType === "reasoning" || rawType === "thinking")
        ? extractReasoningText(record)
        : undefined),
    metadata: { raw: record },
  };
}

function toPartType(type: string): MessagePartType {
  switch (type) {
    case "text":
      return "text";
    case "thinking":
    case "reasoning":
      return "reasoning";
    case "tool_use":
    case "toolUse":
    case "tool-use":
    case "toolCall":
    case "functionCall":
    case "function_call":
    case "function_call_output":
    case "tool_result":
    case "toolResult":
    case "tool":
      return "tool";
    case "patch":
      return "patch";
    case "file":
    case "image":
      return "file";
    case "subtask":
      return "subtask";
    case "compaction":
      return "compaction";
    case "step_start":
    case "step-start":
      return "step_start";
    case "step_finish":
    case "step-finish":
      return "step_finish";
    case "snapshot":
      return "snapshot";
    case "retry":
      return "retry";
    case "agent":
      return "agent";
    default:
      return "agent";
  }
}

/**
 * Convert AgentMessage content into plain text for DB storage.
 *
 * For content block arrays we keep only text blocks to avoid persisting raw
 * JSON syntax that can later pollute assembled model context.
 */
function extractMessageContent(content: unknown): string {
  if (typeof content === "string") {
    return content;
  }

  if (Array.isArray(content)) {
    return content
      .filter((block): block is { type?: unknown; text?: unknown } => {
        return !!block && typeof block === "object";
      })
      .filter((block) => block.type === "text" && typeof block.text === "string")
      .map((block) => block.text as string)
      .join("\n");
  }

  const serialized = JSON.stringify(content);
  return typeof serialized === "string" ? serialized : "";
}

function toRuntimeRoleForTokenEstimate(role: string): "user" | "assistant" | "toolResult" {
  if (role === "tool" || role === "toolResult") {
    return "toolResult";
  }
  if (role === "user" || role === "system") {
    return "user";
  }
  return "assistant";
}

function isTextBlock(value: unknown): value is { type: "text"; text: string } {
  if (!value || typeof value !== "object" || Array.isArray(value)) {
    return false;
  }
  const record = value as Record<string, unknown>;
  return record.type === "text" && typeof record.text === "string";
}

/**
 * Estimate token usage for the content shape that the assembler will emit.
 *
 * LCM stores a plain-text fallback copy in messages.content, but message_parts
 * can rehydrate larger structured/raw blocks. This estimator mirrors the
 * rehydrated shape so compaction decisions use realistic token totals.
 */
function estimateContentTokensForRole(params: {
  role: "user" | "assistant" | "toolResult";
  content: unknown;
  fallbackContent: string;
}): number {
  const { role, content, fallbackContent } = params;

  if (typeof content === "string") {
    return estimateTokens(content);
  }

  if (Array.isArray(content)) {
    if (content.length === 0) {
      return estimateTokens(fallbackContent);
    }

    if (role === "user" && content.length === 1 && isTextBlock(content[0])) {
      return estimateTokens(content[0].text);
    }

    const serialized = JSON.stringify(content);
    return estimateTokens(typeof serialized === "string" ? serialized : "");
  }

  if (content && typeof content === "object") {
    if (role === "user" && isTextBlock(content)) {
      return estimateTokens(content.text);
    }

    const serialized = JSON.stringify([content]);
    return estimateTokens(typeof serialized === "string" ? serialized : "");
  }

  return estimateTokens(fallbackContent);
}

function buildMessageParts(params: {
  sessionId: string;
  message: AgentMessage;
  fallbackContent: string;
}): import("./store/conversation-store.js").CreateMessagePartInput[] {
  const { sessionId, message, fallbackContent } = params;
  const role = typeof message.role === "string" ? message.role : "unknown";
  const topLevel = message as unknown as Record<string, unknown>;
  const topLevelToolCallId =
    safeString(topLevel.toolCallId) ??
    safeString(topLevel.tool_call_id) ??
    safeString(topLevel.toolUseId) ??
    safeString(topLevel.tool_use_id) ??
    safeString(topLevel.call_id) ??
    safeString(topLevel.id);
  const topLevelToolName =
    safeString(topLevel.toolName) ??
    safeString(topLevel.tool_name);
  const topLevelIsError =
    safeBoolean(topLevel.isError) ??
    safeBoolean(topLevel.is_error);

  // BashExecutionMessage: preserve a synthetic text part so output is round-trippable.
  if (!("content" in message) && "command" in message && "output" in message) {
    return [
      {
        sessionId,
        partType: "text",
        ordinal: 0,
        textContent: fallbackContent,
        metadata: toJson({
          originalRole: role,
          source: "bash-exec",
          command: safeString((message as { command?: unknown }).command),
        }),
      },
    ];
  }

  if (!("content" in message)) {
    return [
      {
        sessionId,
        partType: "agent",
        ordinal: 0,
        textContent: fallbackContent || null,
        metadata: toJson({
          originalRole: role,
          source: "unknown-message-shape",
          raw: message,
        }),
      },
    ];
  }

  if (typeof message.content === "string") {
    return [
      {
        sessionId,
        partType: "text",
        ordinal: 0,
        textContent: message.content,
        metadata: toJson({
          originalRole: role,
          toolCallId: topLevelToolCallId,
          toolName: topLevelToolName,
          isError: topLevelIsError,
        }),
      },
    ];
  }

  if (!Array.isArray(message.content)) {
    return [
      {
        sessionId,
        partType: "agent",
        ordinal: 0,
        textContent: fallbackContent || null,
        metadata: toJson({
          originalRole: role,
          source: "non-array-content",
          raw: message.content,
        }),
      },
    ];
  }

  const parts: CreateMessagePartInput[] = [];
  for (let ordinal = 0; ordinal < message.content.length; ordinal++) {
    const block = normalizeUnknownBlock(message.content[ordinal]);
    const metadataRecord = block.metadata.raw as Record<string, unknown> | undefined;
    const partType = toPartType(block.type);
    const toolCallId =
      safeString(metadataRecord?.toolCallId) ??
      safeString(metadataRecord?.tool_call_id) ??
      safeString(metadataRecord?.toolUseId) ??
      safeString(metadataRecord?.tool_use_id) ??
      safeString(metadataRecord?.call_id) ??
      (partType === "tool" ? safeString(metadataRecord?.id) : undefined) ??
      topLevelToolCallId;

    parts.push({
      sessionId,
      partType,
      ordinal,
      textContent: block.text ?? null,
      toolCallId,
      toolName:
        safeString(metadataRecord?.name) ??
        safeString(metadataRecord?.toolName) ??
        safeString(metadataRecord?.tool_name) ??
        topLevelToolName,
      toolInput:
        metadataRecord?.input !== undefined
          ? toJson(metadataRecord.input)
          : metadataRecord?.arguments !== undefined
            ? toJson(metadataRecord.arguments)
          : metadataRecord?.toolInput !== undefined
            ? toJson(metadataRecord.toolInput)
            : (safeString(metadataRecord?.tool_input) ?? null),
      toolOutput:
        metadataRecord?.output !== undefined
          ? toJson(metadataRecord.output)
          : metadataRecord?.toolOutput !== undefined
            ? toJson(metadataRecord.toolOutput)
            : (safeString(metadataRecord?.tool_output) ?? null),
      metadata: toJson({
        originalRole: role,
        toolCallId: topLevelToolCallId,
        toolName: topLevelToolName,
        isError: topLevelIsError,
        rawType: block.type,
        raw: metadataRecord ?? message.content[ordinal],
      }),
    });
  }

  return parts;
}

/**
 * Map AgentMessage role to the DB enum.
 *
 *   "user"      -> "user"
 *   "assistant" -> "assistant"
 *
 * AgentMessage only has user/assistant roles, but we keep the mapping
 * explicit for clarity and future-proofing.
 */
function toDbRole(role: string): "user" | "assistant" | "system" | "tool" {
  if (role === "tool" || role === "toolResult") {
    return "tool";
  }
  if (role === "system") {
    return "system";
  }
  if (role === "user") {
    return "user";
  }
  if (role === "assistant") {
    return "assistant";
  }
  // Unknown roles are preserved via message_parts metadata and treated as assistant.
  return "assistant";
}

type StoredMessage = {
  role: "user" | "assistant" | "system" | "tool";
  content: string;
  tokenCount: number;
};

/**
 * Normalize AgentMessage variants into the storage shape used by LCM.
 */
function toStoredMessage(message: AgentMessage): StoredMessage {
  const content =
    "content" in message
      ? extractMessageContent(message.content)
      : "output" in message
        ? `$ ${(message as { command: string; output: string }).command}\n${(message as { command: string; output: string }).output}`
        : "";
  const runtimeRole = toRuntimeRoleForTokenEstimate(message.role);
  const tokenCount =
    "content" in message
      ? estimateContentTokensForRole({
          role: runtimeRole,
          content: message.content,
          fallbackContent: content,
        })
      : estimateTokens(content);

  return {
    role: toDbRole(message.role),
    content,
    tokenCount,
  };
}

function estimateMessageContentTokensForAfterTurn(content: unknown): number {
  if (typeof content === "string") {
    return estimateTokens(content);
  }
  if (Array.isArray(content)) {
    let total = 0;
    for (const part of content) {
      if (!part || typeof part !== "object") {
        continue;
      }
      const record = part as Record<string, unknown>;
      const text =
        typeof record.text === "string"
          ? record.text
          : typeof record.thinking === "string"
            ? record.thinking
            : "";
      if (text) {
        total += estimateTokens(text);
      }
    }
    return total;
  }
  if (content == null) {
    return 0;
  }
  const serialized = JSON.stringify(content);
  return estimateTokens(typeof serialized === "string" ? serialized : "");
}

function estimateSessionTokenCountForAfterTurn(messages: AgentMessage[]): number {
  let total = 0;
  for (const message of messages) {
    if ("content" in message) {
      total += estimateMessageContentTokensForAfterTurn(message.content);
      continue;
    }
    if ("command" in message || "output" in message) {
      const commandText =
        typeof (message as { command?: unknown }).command === "string"
          ? (message as { command?: string }).command
          : "";
      const outputText =
        typeof (message as { output?: unknown }).output === "string"
          ? (message as { output?: string }).output
          : "";
      total += estimateTokens(`${commandText}\n${outputText}`);
    }
  }
  return total;
}

function isBootstrapMessage(value: unknown): value is AgentMessage {
  if (!value || typeof value !== "object") {
    return false;
  }
  const msg = value as { role?: unknown; content?: unknown; command?: unknown; output?: unknown };
  if (typeof msg.role !== "string") {
    return false;
  }
  return "content" in msg || ("command" in msg && "output" in msg);
}

/** Load recoverable messages from a JSON/JSONL session file. */
function readLeafPathMessages(sessionFile: string): AgentMessage[] {
  let raw = "";
  try {
    raw = readFileSync(sessionFile, "utf8");
  } catch {
    return [];
  }

  const trimmed = raw.trim();
  if (!trimmed) {
    return [];
  }

  if (trimmed.startsWith("[")) {
    try {
      const parsed = JSON.parse(trimmed);
      if (!Array.isArray(parsed)) {
        return [];
      }
      return parsed.filter(isBootstrapMessage);
    } catch {
      return [];
    }
  }

  const messages: AgentMessage[] = [];
  const lines = raw.split(/\r?\n/);
  for (const line of lines) {
    const item = line.trim();
    if (!item) {
      continue;
    }
    try {
      const parsed = JSON.parse(item);
      const candidate =
        parsed && typeof parsed === "object" && "message" in parsed
          ? (parsed as { message?: unknown }).message
          : parsed;
      if (isBootstrapMessage(candidate)) {
        messages.push(candidate);
      }
    } catch {
      // Skip malformed lines.
    }
  }
  return messages;
}

function messageIdentity(role: string, content: string): string {
  return `${role}\u0000${content}`;
}

// ── LcmContextEngine ────────────────────────────────────────────────────────

export class LcmContextEngine implements ContextEngine {
  readonly info: ContextEngineInfo = {
    id: "lcm",
    name: "Lossless Context Management Engine",
    version: "0.1.0",
    ownsCompaction: true,
  };

  private config: LcmConfig;

  /** Get the configured timezone, falling back to system timezone. */
  get timezone(): string {
    return this.config.timezone ?? Intl.DateTimeFormat().resolvedOptions().timeZone;
  }

  private conversationStore: ConversationStore;
  private summaryStore: SummaryStore;
  private assembler: ContextAssembler;
  private compaction: CompactionEngine;
  private retrieval: RetrievalEngine;
  private migrated = false;
  private readonly fts5Available: boolean;
  private sessionOperationQueues = new Map<string, Promise<void>>();
  private largeFileTextSummarizerResolved = false;
  private largeFileTextSummarizer?: (prompt: string) => Promise<string | null>;
  private deps: LcmDependencies;

  constructor(deps: LcmDependencies) {
    this.deps = deps;
    this.config = deps.config;

    const db = getLcmConnection(this.config.databasePath);
    this.fts5Available = getLcmDbFeatures(db).fts5Available;

    this.conversationStore = new ConversationStore(db, { fts5Available: this.fts5Available });
    this.summaryStore = new SummaryStore(db, { fts5Available: this.fts5Available });

    if (!this.fts5Available) {
      this.deps.log.warn(
        "[lcm] FTS5 unavailable in the current Node runtime; full_text search will fall back to LIKE and indexing is disabled",
      );
    }

    this.assembler = new ContextAssembler(
      this.conversationStore,
      this.summaryStore,
      this.config.timezone,
    );

    const compactionConfig: CompactionConfig = {
      contextThreshold: this.config.contextThreshold,
      freshTailCount: this.config.freshTailCount,
      leafMinFanout: this.config.leafMinFanout,
      condensedMinFanout: this.config.condensedMinFanout,
      condensedMinFanoutHard: this.config.condensedMinFanoutHard,
      incrementalMaxDepth: this.config.incrementalMaxDepth,
      leafChunkTokens: this.config.leafChunkTokens,
      leafTargetTokens: this.config.leafTargetTokens,
      condensedTargetTokens: this.config.condensedTargetTokens,
      maxRounds: 10,
      timezone: this.config.timezone,
    };
    this.compaction = new CompactionEngine(
      this.conversationStore,
      this.summaryStore,
      compactionConfig,
    );

    this.retrieval = new RetrievalEngine(this.conversationStore, this.summaryStore);
  }

  /** Ensure DB schema is up-to-date. Called lazily on first bootstrap/ingest/assemble/compact. */
  private ensureMigrated(): void {
    if (this.migrated) {
      return;
    }
    const db = getLcmConnection(this.config.databasePath);
    runLcmMigrations(db, { fts5Available: this.fts5Available });
    this.migrated = true;
  }

  /**
   * Serialize mutating operations per session to prevent ingest/compaction races.
   */
  private async withSessionQueue<T>(sessionId: string, operation: () => Promise<T>): Promise<T> {
    const previous = this.sessionOperationQueues.get(sessionId) ?? Promise.resolve();
    let releaseQueue: () => void = () => {};
    const current = new Promise<void>((resolve) => {
      releaseQueue = resolve;
    });
    const next = previous.catch(() => {}).then(() => current);
    this.sessionOperationQueues.set(sessionId, next);

    await previous.catch(() => {});
    try {
      return await operation();
    } finally {
      releaseQueue();
      void next.finally(() => {
        if (this.sessionOperationQueues.get(sessionId) === next) {
          this.sessionOperationQueues.delete(sessionId);
        }
      });
    }
  }

  /** Normalize optional live token estimates supplied by runtime callers. */
  private normalizeObservedTokenCount(value: unknown): number | undefined {
    if (typeof value !== "number" || !Number.isFinite(value) || value <= 0) {
      return undefined;
    }
    return Math.floor(value);
  }

  /** Format token counts for compact human-readable status strings. */
  private formatTokenCountCompact(value: number | undefined): string {
    if (typeof value !== "number" || !Number.isFinite(value) || value < 0) {
      return "unknown";
    }
    const normalized = Math.floor(value);
    if (normalized >= 1_000_000) {
      const millions = normalized / 1_000_000;
      return `${millions >= 10 ? millions.toFixed(0) : millions.toFixed(1).replace(/\.0$/, "")}m`;
    }
    if (normalized >= 1_000) {
      const thousands = normalized / 1_000;
      return `${thousands >= 10 ? thousands.toFixed(0) : thousands.toFixed(1).replace(/\.0$/, "")}k`;
    }
    return `${normalized}`;
  }

  private async collectCompactionDiagnostics(params: {
    conversationId: number;
    tokenBudget: number;
    decisionCurrentTokens: number;
    thresholdTokens: number;
    targetTokens: number;
    summarizerLabel?: string;
    observedTokens?: number;
    noEligibleSweepChunks?: boolean;
  }): Promise<{
    compactableTokens: number;
    assembledPromptTokens: number;
    totalContextItems: number;
    outsideProtectedTailItems: number;
    protectedTailTokens: number;
    protectedTailItems: number;
    rawTokensUntilLeafTrigger: number;
    summarizerLabel?: string;
    livePromptTokens?: number;
    decisionTokens: number;
    thresholdTokens: number;
    targetTokens: number;
    rawTokensOutsideFreshTail: number;
    leafTriggerThreshold: number;
    noEligibleSweepChunks: boolean;
    sweepBlockerCode?: string;
    sweepBlockerMessage?: string;
  }> {
    const compactableTokens = await this.summaryStore.getContextTokenCount(params.conversationId);
    const leafTrigger = await this.compaction.evaluateLeafTrigger(params.conversationId);
    const assembly = await this.collectAssemblyDiagnostics(params.conversationId, params.tokenBudget);
    const sweepBlocker =
      params.noEligibleSweepChunks === true
        ? await this.diagnoseNoOpSweepBlocker(params.conversationId)
        : undefined;
    return {
      compactableTokens,
      assembledPromptTokens: assembly.assembledPromptTokens,
      totalContextItems: assembly.totalContextItems,
      outsideProtectedTailItems: assembly.outsideProtectedTailItems,
      protectedTailTokens: assembly.protectedTailTokens,
      protectedTailItems: assembly.protectedTailItems,
      rawTokensUntilLeafTrigger: Math.max(0, leafTrigger.threshold - leafTrigger.rawTokensOutsideTail),
      ...(params.summarizerLabel ? { summarizerLabel: params.summarizerLabel } : {}),
      ...(typeof params.observedTokens === "number" ? { livePromptTokens: params.observedTokens } : {}),
      decisionTokens: params.decisionCurrentTokens,
      thresholdTokens: params.thresholdTokens,
      targetTokens: params.targetTokens,
      rawTokensOutsideFreshTail: leafTrigger.rawTokensOutsideTail,
      leafTriggerThreshold: leafTrigger.threshold,
      noEligibleSweepChunks: params.noEligibleSweepChunks === true,
      ...(sweepBlocker ? { sweepBlockerCode: sweepBlocker.code, sweepBlockerMessage: sweepBlocker.message } : {}),
    };
  }

  private buildCompactionReasonText(
    baseReason: string,
    diagnostics: {
      compactableTokens: number;
      assembledPromptTokens: number;
      totalContextItems: number;
      outsideProtectedTailItems: number;
      protectedTailTokens: number;
      protectedTailItems: number;
      rawTokensUntilLeafTrigger: number;
      summarizerLabel?: string;
      livePromptTokens?: number;
      thresholdTokens: number;
      targetTokens: number;
      rawTokensOutsideFreshTail: number;
      leafTriggerThreshold: number;
      noEligibleSweepChunks: boolean;
      sweepBlockerCode?: string;
      sweepBlockerMessage?: string;
    },
  ): string {
    const k = this.formatTokenCountCompact.bind(this);
    const lines: string[] = [];

    const isCompacted = baseReason === "compacted";
    const isFailure = !isCompacted && (baseReason === "could not reach target");

    // Line 1: status
    if (isCompacted) {
      lines.push("⚙️ Compacted");
    } else if (isFailure) {
      lines.push("❌ Compaction failed");
    } else {
      lines.push("⚙️ Compaction skipped");
    }

    // Line 2: token change (only for compacted — caller adds before/after)
    // The caller (OpenClaw core) wraps this with "Compacted (before → after)"
    // so we don't duplicate that here. Instead we show tree info.

    // Line 2: summarizer model
    if (diagnostics.summarizerLabel) {
      lines.push(`🧠 ${diagnostics.summarizerLabel}`);
    }

    // Line 3: tree structure / skip reason
    if (isCompacted) {
      lines.push(`🌿 leaf pass`);
    } else {
      switch (diagnostics.sweepBlockerCode) {
        case "fresh_tail_protected":
          lines.push(`🌿 all items in protected tail`);
          break;
        case "raw_chunk_not_selected":
          lines.push(`🌿 raw msgs outside tail`);
          break;
        case "insufficient_summary_fanout":
          lines.push(`🌿 fanout too small`);
          break;
        case "insufficient_summary_mass":
          lines.push(`🌿 mass too small`);
          break;
        case "no_contiguous_summary_chunk":
          lines.push(`🌿 no eligible chunk`);
          break;
        case "empty_context":
          lines.push(`🌿 context empty`);
          break;
        default:
          if (diagnostics.rawTokensOutsideFreshTail > 0 || diagnostics.leafTriggerThreshold > 0) {
            const outsideTailText = k(diagnostics.rawTokensOutsideFreshTail);
            const triggerText = k(diagnostics.leafTriggerThreshold);
            lines.push(`🌿 raw ${outsideTailText} / trigger at ${triggerText}`);
          }
          break;
      }
    }

    // Protected tail info
    const protectedTailText = k(diagnostics.protectedTailTokens);
    lines.push(`📋 protected ${protectedTailText} · ${diagnostics.protectedTailItems} items`);

    // NOTE: Context usage line is appended by OpenClaw core (formatContextUsageShort),
    // so we don't add our own to avoid duplication.

    return lines.join("\n");
  }

  private resolveLeafChunkTokensForDiagnostics(): number {
    if (
      typeof this.config.leafChunkTokens === "number" &&
      Number.isFinite(this.config.leafChunkTokens) &&
      this.config.leafChunkTokens > 0
    ) {
      return Math.floor(this.config.leafChunkTokens);
    }
    return 20_000;
  }

  private resolveFreshTailCountForDiagnostics(): number {
    if (
      typeof this.config.freshTailCount === "number" &&
      Number.isFinite(this.config.freshTailCount) &&
      this.config.freshTailCount > 0
    ) {
      return Math.floor(this.config.freshTailCount);
    }
    return 8;
  }

  private async resolveContextItemTokenCount(item: {
    itemType: string;
    messageId: number | null;
    summaryId: string | null;
  }): Promise<number> {
    if (item.itemType === "message" && item.messageId != null) {
      const message = await this.conversationStore.getMessageById(item.messageId);
      return message?.tokenCount ?? 0;
    }
    if (item.itemType === "summary" && item.summaryId != null) {
      const summary = await this.summaryStore.getSummary(item.summaryId);
      return summary?.tokenCount ?? 0;
    }
    return 0;
  }

  private async collectAssemblyDiagnostics(
    conversationId: number,
    tokenBudget: number,
  ): Promise<{
    assembledPromptTokens: number;
    totalContextItems: number;
    outsideProtectedTailItems: number;
    protectedTailTokens: number;
    protectedTailItems: number;
  }> {
    const freshTailCount = this.resolveFreshTailCountForDiagnostics();
    const assembled = await this.assembler.assemble({
      conversationId,
      tokenBudget,
      freshTailCount,
    });
    const contextItems = await this.summaryStore.getContextItems(conversationId);
    const tailStart = Math.max(0, contextItems.length - freshTailCount);
    const protectedTail = contextItems.slice(tailStart);

    let protectedTailTokens = 0;
    for (const item of protectedTail) {
      protectedTailTokens += await this.resolveContextItemTokenCount(item);
    }

    return {
      assembledPromptTokens: assembled.estimatedTokens,
      totalContextItems: contextItems.length,
      outsideProtectedTailItems: tailStart,
      protectedTailTokens,
      protectedTailItems: protectedTail.length,
    };
  }

  private resolveFreshTailOrdinalFromContextItems(
    contextItems: Array<{ ordinal: number; itemType: string; messageId: number | null }>,
  ): number {
    const freshTailCount =
      typeof this.config.freshTailCount === "number" &&
      Number.isFinite(this.config.freshTailCount) &&
      this.config.freshTailCount > 0
        ? Math.floor(this.config.freshTailCount)
        : 0;
    if (freshTailCount <= 0) {
      return Infinity;
    }

    const rawMessageItems = contextItems.filter(
      (item) => item.itemType === "message" && item.messageId != null,
    );
    if (rawMessageItems.length === 0) {
      return Infinity;
    }

    const tailStartIdx = Math.max(0, rawMessageItems.length - freshTailCount);
    return rawMessageItems[tailStartIdx]?.ordinal ?? Infinity;
  }

  private resolveFanoutForDepthForDiagnostics(targetDepth: number): number {
    if (targetDepth === 0) {
      if (
        typeof this.config.leafMinFanout === "number" &&
        Number.isFinite(this.config.leafMinFanout) &&
        this.config.leafMinFanout > 0
      ) {
        return Math.floor(this.config.leafMinFanout);
      }
      return 8;
    }
    if (
      typeof this.config.condensedMinFanout === "number" &&
      Number.isFinite(this.config.condensedMinFanout) &&
      this.config.condensedMinFanout > 0
    ) {
      return Math.floor(this.config.condensedMinFanout);
    }
    return 4;
  }

  private resolveCondensedMinChunkTokensForDiagnostics(): number {
    const chunkTarget = this.resolveLeafChunkTokensForDiagnostics();
    const ratioFloor = Math.floor(chunkTarget * 0.1);
    const configuredTarget =
      typeof this.config.condensedTargetTokens === "number" &&
      Number.isFinite(this.config.condensedTargetTokens) &&
      this.config.condensedTargetTokens > 0
        ? Math.floor(this.config.condensedTargetTokens)
        : 0;
    return Math.max(configuredTarget, ratioFloor);
  }

  private async getOldestSummaryChunkAtDepthForDiagnostics(params: {
    conversationId: number;
    targetDepth: number;
    freshTailOrdinal: number;
  }): Promise<{ count: number; summaryTokens: number }> {
    const contextItems = await this.summaryStore.getContextItems(params.conversationId);
    const chunkTokenBudget = this.resolveLeafChunkTokensForDiagnostics();

    let count = 0;
    let summaryTokens = 0;
    for (const item of contextItems) {
      if (item.ordinal >= params.freshTailOrdinal) {
        break;
      }
      if (item.itemType !== "summary" || item.summaryId == null) {
        if (count > 0) {
          break;
        }
        continue;
      }

      const summary = await this.summaryStore.getSummary(item.summaryId);
      if (!summary) {
        if (count > 0) {
          break;
        }
        continue;
      }
      if (summary.depth !== params.targetDepth) {
        if (count > 0) {
          break;
        }
        continue;
      }

      const tokenCount =
        typeof summary.tokenCount === "number" && Number.isFinite(summary.tokenCount) && summary.tokenCount > 0
          ? Math.floor(summary.tokenCount)
          : estimateTokens(summary.content);

      if (count > 0 && summaryTokens + tokenCount > chunkTokenBudget) {
        break;
      }

      count += 1;
      summaryTokens += tokenCount;
      if (summaryTokens >= chunkTokenBudget) {
        break;
      }
    }

    return { count, summaryTokens };
  }

  private async diagnoseNoOpSweepBlocker(conversationId: number): Promise<
    | {
        code: string;
        message: string;
      }
    | undefined
  > {
    const contextItems = await this.summaryStore.getContextItems(conversationId);
    if (contextItems.length === 0) {
      return { code: "empty_context", message: "empty context items" };
    }

    const freshTailOrdinal = this.resolveFreshTailOrdinalFromContextItems(contextItems);
    const rawMessagesOutsideTail = contextItems.filter(
      (item) => item.ordinal < freshTailOrdinal && item.itemType === "message" && item.messageId != null,
    ).length;
    if (rawMessagesOutsideTail > 0) {
      return {
        code: "raw_chunk_not_selected",
        message: `raw messages outside fresh tail still present (${rawMessagesOutsideTail})`,
      };
    }

    const depthLevels = await this.summaryStore.getDistinctDepthsInContext(conversationId, {
      maxOrdinalExclusive: freshTailOrdinal,
    });
    if (depthLevels.length === 0) {
      return {
        code: "fresh_tail_protected",
        message: "fresh tail protects all raw messages and no summary depth is eligible yet",
      };
    }

    const minChunkTokens = this.resolveCondensedMinChunkTokensForDiagnostics();
    for (const targetDepth of depthLevels) {
      const fanout = this.resolveFanoutForDepthForDiagnostics(targetDepth);
      const chunk = await this.getOldestSummaryChunkAtDepthForDiagnostics({
        conversationId,
        targetDepth,
        freshTailOrdinal,
      });
      if (chunk.count < fanout) {
        return {
          code: "insufficient_summary_fanout",
          message: `summary fanout ${chunk.count}/${fanout} at depth ${targetDepth} is too small`,
        };
      }
      if (chunk.summaryTokens < minChunkTokens) {
        return {
          code: "insufficient_summary_mass",
          message: `summary chunk ${this.formatTokenCountCompact(chunk.summaryTokens)}/${this.formatTokenCountCompact(minChunkTokens)} at depth ${targetDepth} is too small`,
        };
      }
    }

    return {
      code: "no_contiguous_summary_chunk",
      message: "no contiguous summary chunk outside protected tail is eligible",
    };
  }

  /** Resolve token budget from direct params or legacy fallback input. */
  private resolveTokenBudget(params: {
    tokenBudget?: number;
    legacyParams?: Record<string, unknown>;
  }): number | undefined {
    const lp = params.legacyParams ?? {};
    if (
      typeof params.tokenBudget === "number" &&
      Number.isFinite(params.tokenBudget) &&
      params.tokenBudget > 0
    ) {
      return Math.floor(params.tokenBudget);
    }
    if (
      typeof lp.tokenBudget === "number" &&
      Number.isFinite(lp.tokenBudget) &&
      lp.tokenBudget > 0
    ) {
      return Math.floor(lp.tokenBudget);
    }
    return undefined;
  }

  /** Resolve an LCM conversation id from a session key via the session store. */
  private async resolveConversationIdForSessionKey(
    sessionKey: string,
  ): Promise<number | undefined> {
    const trimmedKey = sessionKey.trim();
    if (!trimmedKey) {
      return undefined;
    }
    try {
      const runtimeSessionId = await this.deps.resolveSessionIdFromSessionKey(trimmedKey);
      if (!runtimeSessionId) {
        return undefined;
      }
      const conversation =
        await this.conversationStore.getConversationBySessionId(runtimeSessionId);
      return conversation?.conversationId;
    } catch {
      return undefined;
    }
  }

  private resolveSummarizerLabel(legacyParams?: Record<string, unknown>): string | undefined {
    const lp = legacyParams ?? {};
    const runtimeConfig =
      lp.config && typeof lp.config === "object"
        ? (lp.config as {
            summaryModel?: unknown;
            plugins?: {
              entries?: {
                [key: string]: {
                  config?: { summaryModel?: unknown };
                };
              };
            };
          })
        : undefined;
    const nestedPluginSummaryModel =
      typeof runtimeConfig?.plugins?.entries?.["lossless-claw"]?.config?.summaryModel ===
      "string"
        ? runtimeConfig.plugins.entries["lossless-claw"].config.summaryModel.trim()
        : "";
    const summaryModelOverride =
      (typeof runtimeConfig?.summaryModel === "string" ? runtimeConfig.summaryModel.trim() : "") ||
      nestedPluginSummaryModel;
    const providerHint = typeof lp.provider === "string" ? lp.provider.trim() : "";
    const modelHint = typeof lp.model === "string" ? lp.model.trim() : "";
    const modelRef = summaryModelOverride || modelHint || undefined;
    const resolveProviderHint = summaryModelOverride ? undefined : providerHint || undefined;
    try {
      const resolved = this.deps.resolveModel(modelRef, resolveProviderHint);
      if (resolved.provider && resolved.model) {
        return `${resolved.provider}/${resolved.model}`;
      }
    } catch {
      // ignore and fall through to raw hints
    }
    if (summaryModelOverride) {
      return summaryModelOverride;
    }
    if (providerHint && modelHint) {
      return `${providerHint}/${modelHint}`;
    }
    return undefined;
  }

  /** Build a summarize callback with runtime provider fallback handling. */
  private async resolveSummarize(params: {
    legacyParams?: Record<string, unknown>;
    customInstructions?: string;
  }): Promise<(text: string, aggressive?: boolean) => Promise<string>> {
    const lp = params.legacyParams ?? {};
    if (typeof lp.summarize === "function") {
      return lp.summarize as (text: string, aggressive?: boolean) => Promise<string>;
    }
    try {
      const runtimeSummarizer = await createLcmSummarizeFromLegacyParams({
        deps: this.deps,
        legacyParams: lp,
        customInstructions: params.customInstructions,
      });
      if (runtimeSummarizer) {
        return runtimeSummarizer;
      }
      console.error(`[lcm] resolveSummarize: createLcmSummarizeFromLegacyParams returned undefined`);
    } catch (err) {
      console.error(`[lcm] resolveSummarize failed, using emergency fallback:`, err instanceof Error ? err.message : err);
    }
    console.error(`[lcm] resolveSummarize: FALLING BACK TO EMERGENCY TRUNCATION`);
    return createEmergencyFallbackSummarize();
  }

  /**
   * Resolve an optional model-backed summarizer for large text file exploration.
   *
   * This is opt-in via env so ingest remains deterministic and lightweight when
   * no summarization model is configured.
   */
  private async resolveLargeFileTextSummarizer(): Promise<
    ((prompt: string) => Promise<string | null>) | undefined
  > {
    if (this.largeFileTextSummarizerResolved) {
      return this.largeFileTextSummarizer;
    }
    this.largeFileTextSummarizerResolved = true;

    const provider = this.deps.config.largeFileSummaryProvider;
    const model = this.deps.config.largeFileSummaryModel;
    if (!provider || !model) {
      return undefined;
    }

    try {
      const summarize = await createLcmSummarizeFromLegacyParams({
        deps: this.deps,
        legacyParams: { provider, model },
      });
      if (!summarize) {
        return undefined;
      }

      this.largeFileTextSummarizer = async (prompt: string): Promise<string | null> => {
        const summary = await summarize(prompt, false);
        if (typeof summary !== "string") {
          return null;
        }
        const trimmed = summary.trim();
        return trimmed.length > 0 ? trimmed : null;
      };
      return this.largeFileTextSummarizer;
    } catch {
      return undefined;
    }
  }

  /** Persist intercepted large-file text payloads to ~/.openclaw/lcm-files. */
  private async storeLargeFileContent(params: {
    conversationId: number;
    fileId: string;
    extension: string;
    content: string;
  }): Promise<string> {
    const dir = join(homedir(), ".openclaw", "lcm-files", String(params.conversationId));
    await mkdir(dir, { recursive: true });

    const normalizedExtension = params.extension.replace(/[^a-z0-9]/gi, "").toLowerCase() || "txt";
    const filePath = join(dir, `${params.fileId}.${normalizedExtension}`);
    await writeFile(filePath, params.content, "utf8");
    return filePath;
  }

  /**
   * Intercept oversized <file> blocks before persistence and replace them with
   * compact file references backed by large_files records.
   */
  private async interceptLargeFiles(params: {
    conversationId: number;
    content: string;
  }): Promise<{ rewrittenContent: string; fileIds: string[] } | null> {
    const blocks = parseFileBlocks(params.content);
    if (blocks.length === 0) {
      return null;
    }

    const threshold = Math.max(1, this.config.largeFileTokenThreshold);
    const summarizeText = await this.resolveLargeFileTextSummarizer();
    const fileIds: string[] = [];
    const rewrittenSegments: string[] = [];
    let cursor = 0;
    let interceptedAny = false;

    for (const block of blocks) {
      const blockTokens = estimateTokens(block.text);
      if (blockTokens < threshold) {
        continue;
      }

      interceptedAny = true;
      const fileId = `file_${randomUUID().replace(/-/g, "").slice(0, 16)}`;
      const extension = extensionFromNameOrMime(block.fileName, block.mimeType);
      const storageUri = await this.storeLargeFileContent({
        conversationId: params.conversationId,
        fileId,
        extension,
        content: block.text,
      });
      const byteSize = Buffer.byteLength(block.text, "utf8");
      const explorationSummary = await generateExplorationSummary({
        content: block.text,
        fileName: block.fileName,
        mimeType: block.mimeType,
        summarizeText,
      });

      await this.summaryStore.insertLargeFile({
        fileId,
        conversationId: params.conversationId,
        fileName: block.fileName,
        mimeType: block.mimeType,
        byteSize,
        storageUri,
        explorationSummary,
      });

      rewrittenSegments.push(params.content.slice(cursor, block.start));
      rewrittenSegments.push(
        formatFileReference({
          fileId,
          fileName: block.fileName,
          mimeType: block.mimeType,
          byteSize,
          summary: explorationSummary,
        }),
      );
      cursor = block.end;
      fileIds.push(fileId);
    }

    if (!interceptedAny) {
      return null;
    }

    rewrittenSegments.push(params.content.slice(cursor));
    return {
      rewrittenContent: rewrittenSegments.join(""),
      fileIds,
    };
  }

  // ── ContextEngine interface ─────────────────────────────────────────────

  /**
   * Reconcile session-file history with persisted messages and append only the
   * tail that is present in JSONL but missing from LCM.
   */
  private async reconcileSessionTail(params: {
    sessionId: string;
    conversationId: number;
    historicalMessages: AgentMessage[];
  }): Promise<{
    importedMessages: number;
    hasOverlap: boolean;
  }> {
    const { sessionId, conversationId, historicalMessages } = params;
    if (historicalMessages.length === 0) {
      return { importedMessages: 0, hasOverlap: false };
    }

    const latestDbMessage = await this.conversationStore.getLastMessage(conversationId);
    if (!latestDbMessage) {
      return { importedMessages: 0, hasOverlap: false };
    }

    const storedHistoricalMessages = historicalMessages.map((message) => toStoredMessage(message));

    // Fast path: one tail comparison for the common in-sync case.
    const latestHistorical = storedHistoricalMessages[storedHistoricalMessages.length - 1];
    const latestIdentity = messageIdentity(latestDbMessage.role, latestDbMessage.content);
    if (latestIdentity === messageIdentity(latestHistorical.role, latestHistorical.content)) {
      const dbOccurrences = await this.conversationStore.countMessagesByIdentity(
        conversationId,
        latestDbMessage.role,
        latestDbMessage.content,
      );
      let historicalOccurrences = 0;
      for (const stored of storedHistoricalMessages) {
        if (messageIdentity(stored.role, stored.content) === latestIdentity) {
          historicalOccurrences += 1;
        }
      }
      if (dbOccurrences === historicalOccurrences) {
        return { importedMessages: 0, hasOverlap: true };
      }
    }

    // Slow path: walk backward through JSONL to find the most recent anchor
    // message that already exists in LCM, then append everything after it.
    let anchorIndex = -1;
    const historicalIdentityTotals = new Map<string, number>();
    for (const stored of storedHistoricalMessages) {
      const identity = messageIdentity(stored.role, stored.content);
      historicalIdentityTotals.set(identity, (historicalIdentityTotals.get(identity) ?? 0) + 1);
    }

    const historicalIdentityCountsAfterIndex = new Map<string, number>();
    const dbIdentityCounts = new Map<string, number>();
    for (let index = storedHistoricalMessages.length - 1; index >= 0; index--) {
      const stored = storedHistoricalMessages[index];
      const identity = messageIdentity(stored.role, stored.content);
      const seenAfter = historicalIdentityCountsAfterIndex.get(identity) ?? 0;
      const total = historicalIdentityTotals.get(identity) ?? 0;
      const occurrencesThroughIndex = total - seenAfter;
      const exists = await this.conversationStore.hasMessage(
        conversationId,
        stored.role,
        stored.content,
      );
      historicalIdentityCountsAfterIndex.set(identity, seenAfter + 1);
      if (!exists) {
        continue;
      }

      let dbCountForIdentity = dbIdentityCounts.get(identity);
      if (dbCountForIdentity === undefined) {
        dbCountForIdentity = await this.conversationStore.countMessagesByIdentity(
          conversationId,
          stored.role,
          stored.content,
        );
        dbIdentityCounts.set(identity, dbCountForIdentity);
      }

      // Match the same occurrence index as the DB tail so repeated empty
      // tool messages do not anchor against a later, still-missing entry.
      if (dbCountForIdentity !== occurrencesThroughIndex) {
        continue;
      }

      anchorIndex = index;
      break;
    }

    if (anchorIndex < 0) {
      return { importedMessages: 0, hasOverlap: false };
    }
    if (anchorIndex >= historicalMessages.length - 1) {
      return { importedMessages: 0, hasOverlap: true };
    }

    const missingTail = historicalMessages.slice(anchorIndex + 1);
    let importedMessages = 0;
    for (const message of missingTail) {
      const result = await this.ingestSingle({ sessionId, message });
      if (result.ingested) {
        importedMessages += 1;
      }
    }

    return { importedMessages, hasOverlap: true };
  }

  async bootstrap(params: { sessionId: string; sessionFile: string }): Promise<BootstrapResult> {
    this.ensureMigrated();

    const result = await this.withSessionQueue(params.sessionId, async () =>
      this.conversationStore.withTransaction(async () => {
        const conversation = await this.conversationStore.getOrCreateConversation(params.sessionId);
        const conversationId = conversation.conversationId;
        const historicalMessages = readLeafPathMessages(params.sessionFile);

        // First-time import path: no LCM rows yet, so seed directly from the
        // active leaf context snapshot.
        const existingCount = await this.conversationStore.getMessageCount(conversationId);
        if (existingCount === 0) {
          if (historicalMessages.length === 0) {
            await this.conversationStore.markConversationBootstrapped(conversationId);
            return {
              bootstrapped: false,
              importedMessages: 0,
              reason: "no leaf-path messages in session",
            };
          }

          const nextSeq = (await this.conversationStore.getMaxSeq(conversationId)) + 1;
          const bulkInput = historicalMessages.map((message, index) => {
            const stored = toStoredMessage(message);
            return {
              conversationId,
              seq: nextSeq + index,
              role: stored.role,
              content: stored.content,
              tokenCount: stored.tokenCount,
            };
          });

          const inserted = await this.conversationStore.createMessagesBulk(bulkInput);
          await this.summaryStore.appendContextMessages(
            conversationId,
            inserted.map((record) => record.messageId),
          );
          await this.conversationStore.markConversationBootstrapped(conversationId);

          // Prune HEARTBEAT_OK turns from the freshly imported data
          if (this.config.pruneHeartbeatOk) {
            const pruned = await this.pruneHeartbeatOkTurns(conversationId);
            if (pruned > 0) {
              console.error(
                `[lcm] bootstrap: pruned ${pruned} HEARTBEAT_OK messages from conversation ${conversationId}`,
              );
            }
          }

          return {
            bootstrapped: true,
            importedMessages: inserted.length,
          };
        }

        // Existing conversation path: reconcile crash gaps by appending JSONL
        // messages that were never persisted to LCM.
        const reconcile = await this.reconcileSessionTail({
          sessionId: params.sessionId,
          conversationId,
          historicalMessages,
        });

        if (!conversation.bootstrappedAt) {
          await this.conversationStore.markConversationBootstrapped(conversationId);
        }

        if (reconcile.importedMessages > 0) {
          return {
            bootstrapped: true,
            importedMessages: reconcile.importedMessages,
            reason: "reconciled missing session messages",
          };
        }

        if (conversation.bootstrappedAt) {
          return {
            bootstrapped: false,
            importedMessages: 0,
            reason: "already bootstrapped",
          };
        }

        return {
          bootstrapped: false,
          importedMessages: 0,
          reason: reconcile.hasOverlap
            ? "conversation already up to date"
            : "conversation already has messages",
        };
      }),
    );

    // Post-bootstrap pruning: clean HEARTBEAT_OK turns that were already
    // in the DB from prior bootstrap cycles (before pruning was enabled).
    if (this.config.pruneHeartbeatOk && result.bootstrapped === false) {
      try {
        const conversation = await this.conversationStore.getConversationBySessionId(
          params.sessionId,
        );
        if (conversation) {
          const pruned = await this.pruneHeartbeatOkTurns(conversation.conversationId);
          if (pruned > 0) {
            console.error(
              `[lcm] bootstrap: retroactively pruned ${pruned} HEARTBEAT_OK messages from conversation ${conversation.conversationId}`,
            );
          }
        }
      } catch (err) {
        console.error(
          `[lcm] bootstrap: heartbeat pruning failed:`,
          err instanceof Error ? err.message : err,
        );
      }
    }

    return result;
  }

  private async ingestSingle(params: {
    sessionId: string;
    message: AgentMessage;
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    const { sessionId, message, isHeartbeat } = params;
    if (isHeartbeat) {
      return { ingested: false };
    }
    const stored = toStoredMessage(message);

    // Get or create conversation for this session
    const conversation = await this.conversationStore.getOrCreateConversation(sessionId);
    const conversationId = conversation.conversationId;
    let messageForParts: AgentMessage = message;
    if (stored.role === "user") {
      const intercepted = await this.interceptLargeFiles({
        conversationId,
        content: stored.content,
      });
      if (intercepted) {
        stored.content = intercepted.rewrittenContent;
        stored.tokenCount = estimateTokens(stored.content);
        if ("content" in message) {
          messageForParts = {
            ...message,
            content: stored.content,
          } as AgentMessage;
        }
      }
    }

    // Determine next sequence number
    const maxSeq = await this.conversationStore.getMaxSeq(conversationId);
    const seq = maxSeq + 1;

    // Persist the message
    const msgRecord = await this.conversationStore.createMessage({
      conversationId,
      seq,
      role: stored.role,
      content: stored.content,
      tokenCount: stored.tokenCount,
    });
    await this.conversationStore.createMessageParts(
      msgRecord.messageId,
      buildMessageParts({
        sessionId,
        message: messageForParts,
        fallbackContent: stored.content,
      }),
    );

    // Append to context items so assembler can see it
    await this.summaryStore.appendContextMessage(conversationId, msgRecord.messageId);

    return { ingested: true };
  }

  async ingest(params: {
    sessionId: string;
    message: AgentMessage;
    isHeartbeat?: boolean;
  }): Promise<IngestResult> {
    this.ensureMigrated();
    return this.withSessionQueue(params.sessionId, () => this.ingestSingle(params));
  }

  async ingestBatch(params: {
    sessionId: string;
    messages: AgentMessage[];
    isHeartbeat?: boolean;
  }): Promise<IngestBatchResult> {
    this.ensureMigrated();
    if (params.messages.length === 0) {
      return { ingestedCount: 0 };
    }
    return this.withSessionQueue(params.sessionId, async () => {
      let ingestedCount = 0;
      for (const message of params.messages) {
        const result = await this.ingestSingle({
          sessionId: params.sessionId,
          message,
          isHeartbeat: params.isHeartbeat,
        });
        if (result.ingested) {
          ingestedCount += 1;
        }
      }
      return { ingestedCount };
    });
  }

  async afterTurn(params: {
    sessionId: string;
    sessionFile: string;
    messages: AgentMessage[];
    prePromptMessageCount: number;
    autoCompactionSummary?: string;
    isHeartbeat?: boolean;
    tokenBudget?: number;
    legacyCompactionParams?: Record<string, unknown>;
  }): Promise<void> {
    this.ensureMigrated();

    const ingestBatch: AgentMessage[] = [];
    if (params.autoCompactionSummary) {
      ingestBatch.push({
        role: "user",
        content: params.autoCompactionSummary,
      } as AgentMessage);
    }

    const newMessages = params.messages.slice(params.prePromptMessageCount);
    ingestBatch.push(...newMessages);
    if (ingestBatch.length === 0) {
      return;
    }

    try {
      await this.ingestBatch({
        sessionId: params.sessionId,
        messages: ingestBatch,
        isHeartbeat: params.isHeartbeat === true,
      });
    } catch {
      // Continue with proactive compaction even if ingest fails.
    }

    const tokenBudget =
      typeof params.tokenBudget === "number" &&
      Number.isFinite(params.tokenBudget) &&
      params.tokenBudget > 0
        ? Math.floor(params.tokenBudget)
        : undefined;
    if (!tokenBudget) {
      return;
    }

    const liveContextTokens = estimateSessionTokenCountForAfterTurn(params.messages);

    try {
      const leafTrigger = await this.evaluateLeafTrigger(params.sessionId);
      if (leafTrigger.shouldCompact) {
        this.compactLeafAsync({
          sessionId: params.sessionId,
          sessionFile: params.sessionFile,
          tokenBudget,
          currentTokenCount: liveContextTokens,
          legacyParams: params.legacyCompactionParams,
        }).catch(() => {
          // Leaf compaction is best-effort and should not fail the caller.
        });
      }
    } catch {
      // Leaf trigger checks are best-effort.
    }

    try {
      await this.compact({
        sessionId: params.sessionId,
        sessionFile: params.sessionFile,
        tokenBudget,
        currentTokenCount: liveContextTokens,
        compactionTarget: "threshold",
        legacyParams: params.legacyCompactionParams,
      });
    } catch {
      // Proactive compaction is best-effort in the post-turn lifecycle.
    }
  }

  async assemble(params: {
    sessionId: string;
    messages: AgentMessage[];
    tokenBudget?: number;
  }): Promise<AssembleResult> {
    try {
      this.ensureMigrated();

      const conversation = await this.conversationStore.getConversationBySessionId(
        params.sessionId,
      );
      if (!conversation) {
        return {
          messages: params.messages,
          estimatedTokens: 0,
        };
      }

      const contextItems = await this.summaryStore.getContextItems(conversation.conversationId);
      if (contextItems.length === 0) {
        return {
          messages: params.messages,
          estimatedTokens: 0,
        };
      }

      // Guard against incomplete bootstrap/coverage: if the DB only has
      // raw context items and clearly trails the current live history, keep
      // the live path to avoid dropping prompt context.
      const hasSummaryItems = contextItems.some((item) => item.itemType === "summary");
      if (!hasSummaryItems && contextItems.length < params.messages.length) {
        return {
          messages: params.messages,
          estimatedTokens: 0,
        };
      }

      const tokenBudget =
        typeof params.tokenBudget === "number" &&
        Number.isFinite(params.tokenBudget) &&
        params.tokenBudget > 0
          ? Math.floor(params.tokenBudget)
          : 128_000;

      const assembled = await this.assembler.assemble({
        conversationId: conversation.conversationId,
        tokenBudget,
        freshTailCount: this.config.freshTailCount,
      });

      // If assembly produced no messages for a non-empty live session,
      // fail safe to the live context.
      if (assembled.messages.length === 0 && params.messages.length > 0) {
        return {
          messages: params.messages,
          estimatedTokens: 0,
        };
      }

      const result: AssembleResultWithSystemPrompt = {
        messages: assembled.messages,
        estimatedTokens: assembled.estimatedTokens,
        ...(assembled.systemPromptAddition
          ? { systemPromptAddition: assembled.systemPromptAddition }
          : {}),
      };
      return result;
    } catch {
      return {
        messages: params.messages,
        estimatedTokens: 0,
      };
    }
  }

  /** Evaluate whether incremental leaf compaction should run for a session. */
  async evaluateLeafTrigger(sessionId: string): Promise<{
    shouldCompact: boolean;
    rawTokensOutsideTail: number;
    threshold: number;
  }> {
    this.ensureMigrated();
    const conversation = await this.conversationStore.getConversationBySessionId(sessionId);
    if (!conversation) {
      const fallbackThreshold =
        typeof this.config.leafChunkTokens === "number" &&
        Number.isFinite(this.config.leafChunkTokens) &&
        this.config.leafChunkTokens > 0
          ? Math.floor(this.config.leafChunkTokens)
          : 20_000;
      return {
        shouldCompact: false,
        rawTokensOutsideTail: 0,
        threshold: fallbackThreshold,
      };
    }
    return this.compaction.evaluateLeafTrigger(conversation.conversationId);
  }

  /** Run one incremental leaf compaction pass in the per-session queue. */
  async compactLeafAsync(params: {
    sessionId: string;
    sessionFile: string;
    tokenBudget?: number;
    currentTokenCount?: number;
    customInstructions?: string;
    legacyParams?: Record<string, unknown>;
    force?: boolean;
    previousSummaryContent?: string;
  }): Promise<CompactResult> {
    this.ensureMigrated();
    return this.withSessionQueue(params.sessionId, async () => {
      const sessionId = params.sessionId;
      const conversation = await this.conversationStore.getConversationBySessionId(
        params.sessionId,
      );
      if (!conversation) {
        return {
          ok: true,
          compacted: false,
          reason: "no conversation found for session",
        };
      }

      const tokenBudget = this.resolveTokenBudget(params);
      if (!tokenBudget) {
        return {
          ok: false,
          compacted: false,
          reason: "missing token budget in compact params",
        };
      }

      const lp = params.legacyParams ?? {};
      const observedTokens = this.normalizeObservedTokenCount(
        params.currentTokenCount ??
          (
            lp as {
              currentTokenCount?: unknown;
            }
          ).currentTokenCount,
      );
      const summarize = await this.resolveSummarize({
        legacyParams: params.legacyParams,
        customInstructions: params.customInstructions,
      });
      const summarizerLabel = this.resolveSummarizerLabel(params.legacyParams);

      const leafResult = await this.compaction.compactLeaf({
        conversationId: conversation.conversationId,
        tokenBudget,
        summarize,
        force: params.force,
        previousSummaryContent: params.previousSummaryContent,
      });
      const thresholdTokens = Math.floor(this.config.contextThreshold * tokenBudget);
      const diagnostics = await this.collectCompactionDiagnostics({
        conversationId: conversation.conversationId,
        tokenBudget,
        decisionCurrentTokens: observedTokens ?? leafResult.tokensBefore,
        thresholdTokens,
        targetTokens: thresholdTokens,
        summarizerLabel,
        observedTokens,
      });
      const tokensBefore = observedTokens ?? leafResult.tokensBefore;

      if (leafResult.actionTaken) {
        this.deps.log.info(
          `[lcm] compact success session=${sessionId} mode=leaf summarizer=${summarizerLabel ?? "unknown"} before=${tokensBefore} after=${leafResult.tokensAfter} assembled=${diagnostics.assembledPromptTokens} protected_tail=${diagnostics.protectedTailTokens}/${diagnostics.protectedTailItems}`,
        );
      }

      return {
        ok: true,
        compacted: leafResult.actionTaken,
        reason: leafResult.actionTaken
          ? this.buildCompactionReasonText("compacted", diagnostics)
          : this.buildCompactionReasonText("below threshold", diagnostics),
        result: {
          tokensBefore,
          tokensAfter: leafResult.tokensAfter,
          details: {
            rounds: leafResult.actionTaken ? 1 : 0,
            targetTokens: tokenBudget,
            mode: "leaf",
            diagnostics,
          },
        },
      };
    });
  }

  async compact(params: {
    sessionId: string;
    sessionFile: string;
    tokenBudget?: number;
    currentTokenCount?: number;
    compactionTarget?: "budget" | "threshold";
    customInstructions?: string;
    legacyParams?: Record<string, unknown>;
    /** Force compaction even if below threshold */
    force?: boolean;
  }): Promise<CompactResult> {
    this.ensureMigrated();
    return this.withSessionQueue(params.sessionId, async () => {
      const { sessionId, force = false } = params;

      // Look up conversation
      const conversation = await this.conversationStore.getConversationBySessionId(sessionId);
      if (!conversation) {
        return {
          ok: true,
          compacted: false,
          reason: "no conversation found for session",
        };
      }

      const conversationId = conversation.conversationId;

      const lp = params.legacyParams ?? {};
      const manualCompactionRequested =
        (
          lp as {
            manualCompaction?: unknown;
          }
        ).manualCompaction === true;
      const forceCompaction = force || manualCompactionRequested;
      const tokenBudget = this.resolveTokenBudget(params);
      if (!tokenBudget) {
        return {
          ok: false,
          compacted: false,
          reason: "missing token budget in compact params",
        };
      }

      const summarize = await this.resolveSummarize({
        legacyParams: params.legacyParams,
        customInstructions: params.customInstructions,
      });
      const summarizerLabel = this.resolveSummarizerLabel(params.legacyParams);

      // Evaluate whether compaction is needed (unless forced)
      const observedTokens = this.normalizeObservedTokenCount(
        params.currentTokenCount ??
          (
            lp as {
              currentTokenCount?: unknown;
            }
          ).currentTokenCount,
      );
      const decision =
        observedTokens !== undefined
          ? await this.compaction.evaluate(conversationId, tokenBudget, observedTokens)
          : await this.compaction.evaluate(conversationId, tokenBudget);
      const targetTokens =
        params.compactionTarget === "threshold" ? decision.threshold : tokenBudget;
      const liveContextStillExceedsTarget =
        observedTokens !== undefined && observedTokens >= targetTokens;

      if (!forceCompaction && !decision.shouldCompact) {
        const diagnostics = await this.collectCompactionDiagnostics({
          conversationId,
          tokenBudget,
          decisionCurrentTokens: decision.currentTokens,
          thresholdTokens: decision.threshold,
          targetTokens,
          summarizerLabel,
          observedTokens,
        });
        return {
          ok: true,
          compacted: false,
          reason: this.buildCompactionReasonText("below threshold", diagnostics),
          result: {
            tokensBefore: decision.currentTokens,
            details: {
              targetTokens,
              diagnostics,
            },
          },
        };
      }

      const useSweep =
        manualCompactionRequested || forceCompaction || params.compactionTarget === "threshold";
      if (useSweep) {
        const sweepResult = await this.compaction.compactFullSweep({
          conversationId,
          tokenBudget,
          summarize,
          force: forceCompaction,
          hardTrigger: false,
        });
        const diagnostics = await this.collectCompactionDiagnostics({
          conversationId,
          tokenBudget,
          decisionCurrentTokens: decision.currentTokens,
          thresholdTokens: decision.threshold,
          targetTokens,
          summarizerLabel,
          observedTokens,
          noEligibleSweepChunks: !sweepResult.actionTaken,
        });
        const skipReason = manualCompactionRequested
          ? "nothing to compact"
          : liveContextStillExceedsTarget
            ? "live context still exceeds target"
            : "already under target";

        if (sweepResult.actionTaken) {
          this.deps.log.info(
            `[lcm] compact success session=${sessionId} mode=sweep summarizer=${summarizerLabel ?? "unknown"} before=${decision.currentTokens} after=${sweepResult.tokensAfter} assembled=${diagnostics.assembledPromptTokens} protected_tail=${diagnostics.protectedTailTokens}/${diagnostics.protectedTailItems}`,
          );
        }

        return {
          ok: sweepResult.actionTaken || !liveContextStillExceedsTarget,
          compacted: sweepResult.actionTaken,
          reason: sweepResult.actionTaken
            ? this.buildCompactionReasonText("compacted", diagnostics)
            : this.buildCompactionReasonText(skipReason, diagnostics),
          result: {
            tokensBefore: decision.currentTokens,
            tokensAfter: sweepResult.tokensAfter,
            details: {
              rounds: sweepResult.actionTaken ? 1 : 0,
              targetTokens,
              diagnostics,
            },
          },
        };
      }

      // When forced, use the token budget as target
      const convergenceTargetTokens = forceCompaction
        ? tokenBudget
        : params.compactionTarget === "threshold"
          ? decision.threshold
          : tokenBudget;

      const compactResult = await this.compaction.compactUntilUnder({
        conversationId,
        tokenBudget,
        targetTokens: convergenceTargetTokens,
        ...(observedTokens !== undefined ? { currentTokens: observedTokens } : {}),
        summarize,
      });
      const didCompact = compactResult.rounds > 0;

      const diagnostics = await this.collectCompactionDiagnostics({
        conversationId,
        tokenBudget,
        decisionCurrentTokens: decision.currentTokens,
        thresholdTokens: decision.threshold,
        targetTokens: convergenceTargetTokens,
        summarizerLabel,
        observedTokens,
        noEligibleSweepChunks: !didCompact,
      });

      if (compactResult.success && didCompact) {
        this.deps.log.info(
          `[lcm] compact success session=${sessionId} mode=converge summarizer=${summarizerLabel ?? "unknown"} before=${decision.currentTokens} after=${compactResult.finalTokens} assembled=${diagnostics.assembledPromptTokens} protected_tail=${diagnostics.protectedTailTokens}/${diagnostics.protectedTailItems}`,
        );
      }

      return {
        ok: compactResult.success,
        compacted: didCompact,
        reason: compactResult.success
          ? didCompact
            ? this.buildCompactionReasonText("compacted", diagnostics)
            : this.buildCompactionReasonText("already under target", diagnostics)
          : this.buildCompactionReasonText("could not reach target", diagnostics),
        result: {
          tokensBefore: decision.currentTokens,
          tokensAfter: compactResult.finalTokens,
          details: {
            rounds: compactResult.rounds,
            targetTokens: convergenceTargetTokens,
            diagnostics,
          },
        },
      };
    });
  }

  /**
   * Return tree status for a specific conversation.
   * Used by the /lcm plugin command.
   */
  async getTreeStatus(conversationId: number): Promise<{
    contextItems: { type: string; count: number }[];
    depthStats: { depth: number; count: number; totalTokens: number; totalDescendants: number; totalDescTokens: number }[];
    totalMessages: number;
    recentSummaries: { kind: string; depth: number; tokenCount: number; sourceTokenCount: number; createdAt: string }[];
  }> {
    const { DatabaseSync } = await import("node:sqlite") as typeof import("node:sqlite");
    const db = new DatabaseSync(this.config.databasePath, { readonly: true });

    try {
      const contextItems = db
        .prepare(
          "SELECT item_type as type, COUNT(*) as count FROM context_items WHERE conversation_id = ? GROUP BY item_type",
        )
        .all(conversationId) as { type: string; count: number }[];

      const depthStats = db
        .prepare(
          `SELECT depth, COUNT(*) as count, SUM(token_count) as totalTokens,
                  SUM(descendant_count) as totalDescendants,
                  SUM(descendant_token_count) as totalDescTokens
           FROM summaries WHERE conversation_id = ?
           GROUP BY depth ORDER BY depth`,
        )
        .all(conversationId) as { depth: number; count: number; totalTokens: number; totalDescendants: number; totalDescTokens: number }[];

      const totalRow = db
        .prepare("SELECT COUNT(*) as totalMessages FROM messages WHERE conversation_id = ?")
        .get(conversationId) as { totalMessages: number } | undefined;

      const recentSummaries = db
        .prepare(
          `SELECT kind, depth, token_count as tokenCount,
                  source_message_token_count as sourceTokenCount, created_at as createdAt
           FROM summaries WHERE conversation_id = ?
           ORDER BY created_at DESC LIMIT 5`,
        )
        .all(conversationId) as { kind: string; depth: number; tokenCount: number; sourceTokenCount: number; createdAt: string }[];

      return {
        contextItems,
        depthStats,
        totalMessages: totalRow?.totalMessages ?? 0,
        recentSummaries,
      };
    } finally {
      db.close();
    }
  }

  async prepareSubagentSpawn(params: {
    parentSessionKey: string;
    childSessionKey: string;
    ttlMs?: number;
  }): Promise<SubagentSpawnPreparation | undefined> {
    this.ensureMigrated();

    const childSessionKey = params.childSessionKey.trim();
    const parentSessionKey = params.parentSessionKey.trim();
    if (!childSessionKey || !parentSessionKey) {
      return undefined;
    }

    const conversationId = await this.resolveConversationIdForSessionKey(parentSessionKey);
    if (typeof conversationId !== "number") {
      return undefined;
    }

    const ttlMs =
      typeof params.ttlMs === "number" && Number.isFinite(params.ttlMs) && params.ttlMs > 0
        ? Math.floor(params.ttlMs)
        : undefined;

    createDelegatedExpansionGrant({
      delegatedSessionKey: childSessionKey,
      issuerSessionId: parentSessionKey,
      allowedConversationIds: [conversationId],
      tokenCap: this.config.maxExpandTokens,
      ttlMs,
    });

    return {
      rollback: () => {
        revokeDelegatedExpansionGrantForSession(childSessionKey, { removeBinding: true });
      },
    };
  }

  async onSubagentEnded(params: {
    childSessionKey: string;
    reason: SubagentEndReason;
  }): Promise<void> {
    const childSessionKey = params.childSessionKey.trim();
    if (!childSessionKey) {
      return;
    }

    switch (params.reason) {
      case "deleted":
        revokeDelegatedExpansionGrantForSession(childSessionKey, { removeBinding: true });
        break;
      case "completed":
        revokeDelegatedExpansionGrantForSession(childSessionKey);
        break;
      case "released":
      case "swept":
        removeDelegatedExpansionGrantForSession(childSessionKey);
        break;
    }
  }

  async dispose(): Promise<void> {
    // No-op for plugin singleton — the connection is shared across runs.
    // OpenClaw's runner calls dispose() after every run, but the plugin
    // registers a single engine instance reused by the factory. Closing
    // the DB here would break subsequent runs with "database is not open".
    // The connection is cleaned up on process exit via closeLcmConnection().
  }

  // ── Public accessors for retrieval (used by subagent expansion) ─────────

  getRetrieval(): RetrievalEngine {
    return this.retrieval;
  }

  getConversationStore(): ConversationStore {
    return this.conversationStore;
  }

  getSummaryStore(): SummaryStore {
    return this.summaryStore;
  }

  // ── Heartbeat pruning ──────────────────────────────────────────────────

  /**
   * Detect HEARTBEAT_OK turn cycles in a conversation and delete them.
   *
   * A HEARTBEAT_OK turn is: a user message (the heartbeat prompt), followed by
   * any tool call/result messages, ending with an assistant message that is a
   * heartbeat ack. The entire sequence has no durable information value for LCM.
   *
   * Detection: assistant content (trimmed, lowercased) starts with "heartbeat_ok"
   * and any text after is not alphanumeric (matches OpenClaw core's ack detection).
   * This catches both exact "HEARTBEAT_OK" and chatty variants like
   * "HEARTBEAT_OK — weekend, no market".
   *
   * Returns the number of messages deleted.
   */
  private async pruneHeartbeatOkTurns(conversationId: number): Promise<number> {
    const allMessages = await this.conversationStore.getMessages(conversationId);
    if (allMessages.length === 0) {
      return 0;
    }

    const toDelete: number[] = [];

    // Walk through messages finding HEARTBEAT_OK assistant replies, then
    // collect the entire turn (back to the preceding user message).
    for (let i = 0; i < allMessages.length; i++) {
      const msg = allMessages[i];
      if (msg.role !== "assistant") {
        continue;
      }
      if (!isHeartbeatOkContent(msg.content)) {
        continue;
      }

      // Found a HEARTBEAT_OK reply. Walk backward to find the turn start
      // (the preceding user message).
      const turnMessageIds: number[] = [msg.messageId];
      for (let j = i - 1; j >= 0; j--) {
        const prev = allMessages[j];
        turnMessageIds.push(prev.messageId);
        if (prev.role === "user") {
          break; // Found turn start
        }
      }

      toDelete.push(...turnMessageIds);
    }

    if (toDelete.length === 0) {
      return 0;
    }

    // Deduplicate (a message could theoretically appear in multiple turns)
    const uniqueIds = [...new Set(toDelete)];
    return this.conversationStore.deleteMessages(uniqueIds);
  }
}

// ── Heartbeat detection ─────────────────────────────────────────────────────

const HEARTBEAT_OK_TOKEN = "heartbeat_ok";

/**
 * Detect whether an assistant message is a heartbeat ack.
 *
 * Matches the same pattern as OpenClaw core's heartbeat-events-filter:
 * content starts with "heartbeat_ok" (case-insensitive) and any character
 * immediately after is not alphanumeric or underscore.
 *
 * This catches:
 *   - "HEARTBEAT_OK"
 *   - "  HEARTBEAT_OK  "
 *   - "HEARTBEAT_OK — weekend, no market."
 *   - "Saturday 10:48 AM PT — weekend, no market. HEARTBEAT_OK"
 *
 * But not:
 *   - "HEARTBEAT_OK_EXTENDED" (alphanumeric continuation)
 */
function isHeartbeatOkContent(content: string): boolean {
  const trimmed = content.trim().toLowerCase();
  if (!trimmed) {
    return false;
  }

  // Check if it starts with the token
  if (trimmed.startsWith(HEARTBEAT_OK_TOKEN)) {
    const suffix = trimmed.slice(HEARTBEAT_OK_TOKEN.length);
    if (suffix.length === 0) {
      return true;
    }
    return !/[a-z0-9_]/.test(suffix[0]);
  }

  // Also check if it ends with the token (chatty prefix + HEARTBEAT_OK)
  if (trimmed.endsWith(HEARTBEAT_OK_TOKEN)) {
    return true;
  }

  return false;
}

// ── Emergency fallback summarization ────────────────────────────────────────

/**
 * Creates a deterministic truncation summarizer used only as an emergency
 * fallback when the model-backed summarizer cannot be created.
 *
 * CompactionEngine already escalates normal -> aggressive -> fallback for
 * convergence. This function simply provides a stable baseline summarize
 * callback to keep compaction operable when runtime setup is unavailable.
 */
function createEmergencyFallbackSummarize(): (
  text: string,
  aggressive?: boolean,
) => Promise<string> {
  return async (text: string, aggressive?: boolean): Promise<string> => {
    const maxChars = aggressive ? 600 * 4 : 900 * 4;
    if (text.length <= maxChars) {
      return text;
    }
    return text.slice(0, maxChars) + "\n[Truncated for context management]";
  };
}
