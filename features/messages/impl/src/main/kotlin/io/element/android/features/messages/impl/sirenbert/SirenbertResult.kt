/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

/**
 * Per-message SIRENBERT classification result for a single Matrix event.
 *
 * Lives in the in-memory [SirenbertCache] keyed by Matrix event id. Phase A.2
 * is in-memory only; persistence (Room) lands in Phase A.3.
 */
data class SirenbertResult(
    /** The role the API saw this message under: "S" suspect or "T" target. */
    val role: String,
    /** Phase A.2 lifecycle of the classification request. */
    val status: Status,
    /** Conversation-level verdict from the API: NON_SCAM / SUSPICIOUS / SCAM. */
    val verdict: String? = null,
    /** Stage 1 trigger label, when status == [Status.Classified]. */
    val trigger: String? = null,
    /** FSM state after this message: NULL/IC/RB/PE/EX. */
    val state: String? = null,
    /** Stage 2 SUSPICIOUS probability, when available. */
    val suspProb: Float? = null,
    /** Stage 2 SCAM probability, when available. */
    val scamProb: Float? = null,
    /** Failure message when status == [Status.Error]. */
    val errorMessage: String? = null,
    /**
     * SIRENBERT cross-app handoff (EMIT side): set to a
     * sirenbert://track?conv=<conversationKey> deeplink when this message
     * classifies as Platform_Migration, so the UI can offer to continue tracking
     * in a second SIRENBERT-enabled client. Null otherwise.
     */
    val handoffUri: String? = null,
) {
    enum class Status {
        /** Cache miss: not yet requested. */
        Idle,

        /** Request dispatched, no response yet. */
        InFlight,

        /** Classified (suspect message). */
        Classified,

        /** Target message used as context only; no Stage 1 vector. */
        ContextOnly,

        /** API errored or network failed. */
        Error,
    }
}
