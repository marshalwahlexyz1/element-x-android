/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import kotlinx.coroutines.flow.Flow
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import kotlinx.coroutines.flow.map

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
}
