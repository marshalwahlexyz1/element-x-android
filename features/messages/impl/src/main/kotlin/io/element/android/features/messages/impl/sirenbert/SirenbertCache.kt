/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.SupervisorJob
import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map
import kotlinx.coroutines.launch
import timber.log.Timber

/**
 * Phase A.2 in-memory cache for SIRENBERT classification results.
 *
 * Keyed by the Matrix event id (the durable "$abc:server" form, NOT the
 * transient TimelineItem.Event.id which is "live28"-style).
 *
 * Singleton because the cache survives composable lifecycle. Persistence
 * (Room) lands in Phase A.3.
 */
object SirenbertCache {
    private val _results = MutableStateFlow<Map<String, SirenbertResult>>(emptyMap())

    /** Whole-cache snapshot, useful for diagnostic flows. */
    val results: Flow<Map<String, SirenbertResult>> = _results.asStateFlow()

    /** Per-event observation. Emits null until something is cached for [eventId]. */
    fun observe(eventId: String): Flow<SirenbertResult?> =
        _results.map { it[eventId] }

    /** Read the current value without subscribing. */
    fun peek(eventId: String): SirenbertResult? = _results.value[eventId]

    /** Insert or overwrite. */
    fun put(eventId: String, result: SirenbertResult) {
        _results.value = _results.value + (eventId to result)
    }

    /** Wipe everything. For development only. */
    fun clear() {
        _results.value = emptyMap()
    }

    /**
     * Inflight de-dup guard. Without this, every recomposition that finds
     * status==Idle would fire a fresh POST. We mark the entry as InFlight
     * immediately so a second caller for the same eventId sees the marker
     * and does nothing.
     */
    private val scope = CoroutineScope(SupervisorJob() + Dispatchers.IO)

    /**
     * Dispatch a /predict POST if (and only if) the cache has no entry for
     * [eventId] yet. Safe to call from any composable's LaunchedEffect.
     *
     * Role "S" means the API will classify the message; "T" means context-only.
     * [body] is the plaintext message; it's sent to the laptop server but never
     * persisted on the phone and never logged (only ids/role are logged).
     */
    fun ensureClassified(
        roomId: String,
        eventId: String,
        role: String,
        body: String,
    ) {
        if (eventId.isEmpty()) return
        val existing = _results.value[eventId]
        if (existing != null && existing.status != SirenbertResult.Status.Idle) return
        // Mark InFlight so concurrent recompositions don't refire.
        put(eventId, SirenbertResult(role = role, status = SirenbertResult.Status.InFlight))
        scope.launch {
            try {
                val resp = SirenbertApiClient.predict(roomId, eventId, role, body)
                val verdict = resp.verdict?.uppercase().orEmpty()
                val isContextOnly = verdict == "CONTEXT_ONLY"
                val result = SirenbertResult(
                    role = role,
                    status = if (isContextOnly) {
                        SirenbertResult.Status.ContextOnly
                    } else {
                        SirenbertResult.Status.Classified
                    },
                    verdict = resp.verdict?.takeIf { it.isNotBlank() && !isContextOnly },
                    trigger = resp.messageTrigger,
                    state = resp.fsmState,
                    suspProb = resp.suspiciousProbability,
                    scamProb = resp.scamProbability,
                )
                put(eventId, result)
                Timber.tag("SIRENBERT").d(
                    "cache put id=%s status=%s trigger=%s state=%s susp=%s scam=%s",
                    eventId.take(12),
                    result.status,
                    result.trigger,
                    result.state,
                    result.suspProb,
                    result.scamProb,
                )
            } catch (t: Throwable) {
                put(
                    eventId,
                    SirenbertResult(
                        role = role,
                        status = SirenbertResult.Status.Error,
                        errorMessage = t.message ?: t::class.java.simpleName,
                    ),
                )
            }
        }
    }
}
