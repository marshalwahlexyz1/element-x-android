/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import android.net.Uri

/**
 * SIRENBERT cross-app handoff deeplink (minimal, dev-facing).
 *
 * A conversation being tracked in one Matrix client can be handed off to a
 * SECOND client that also embeds the SIRENBERT engine, continuing the SAME
 * per-conversation tracking state. Only the opaque [conversationKey]
 * (e.g. "!room:server#s2") crosses apps — never any message body.
 *
 * EMIT side (Element X): when a message classifies as Platform_Migration,
 * surface [trackUri] so the user can jump to the other app.
 * RECEIVE side (SchildiChat): [SirenbertTrackDeeplinkActivity] parses the
 * `conv` query param and seeds the on-device engine buffer via
 * [OnDeviceSirenbertEngine.seedConversation].
 *
 * URI shape: sirenbert://track?conv=<conversationKey>
 */
object SirenbertHandoff {
    const val SCHEME = "sirenbert"
    const val HOST = "track"
    const val PARAM_CONV = "conv"

    /** The Stage-1 trigger that motivates a handoff prompt on the emit side. */
    const val TRIGGER_PLATFORM_MIGRATION = "Platform_Migration"

    /** Build sirenbert://track?conv=<conversationKey>. */
    fun trackUri(conversationKey: String): String =
        Uri.Builder()
            .scheme(SCHEME)
            .authority(HOST)
            .appendQueryParameter(PARAM_CONV, conversationKey)
            .build()
            .toString()

    /** Extract the conversationKey from a sirenbert://track?conv=... uri, or null. */
    fun parseConv(uri: Uri): String? {
        if (uri.scheme != SCHEME || uri.host != HOST) return null
        return uri.getQueryParameter(PARAM_CONV)?.takeIf { it.isNotBlank() }
    }
}
