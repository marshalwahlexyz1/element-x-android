/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import java.util.concurrent.atomic.AtomicInteger

/**
 * Session counter used to build the FastAPI conversation_id.
 *
 * The same Matrix room can host multiple "test conversations" during
 * development; each time the user resets, we bump the counter so the FastAPI
 * server treats it as a brand new conversation and rebuilds Stage 2 hidden
 * state from scratch.
 *
 * The on-the-wire conversation_id is `${roomId}#s${counter}`.
 */
object SirenbertSession {
    private val counter = AtomicInteger(1)

    /** Current session counter (>= 1). */
    val current: Int get() = counter.get()

    /**
     * Build the conversation_id that goes to the FastAPI server for this
     * room in this session.
     */
    fun conversationId(roomId: String): String = "$roomId#s${current}"

    /** Increment the counter; called when the user resets via adb broadcast. */
    fun bump(): Int = counter.incrementAndGet()
}
