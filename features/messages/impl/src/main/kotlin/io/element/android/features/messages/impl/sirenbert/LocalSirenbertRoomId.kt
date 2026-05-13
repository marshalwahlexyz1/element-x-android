/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import androidx.compose.runtime.compositionLocalOf

/**
 * CompositionLocal carrying the current room's Matrix id ("!abc:server.org")
 * down to the SIRENBERT badge composables. Set at the timeline-screen level
 * (MessagesNode / MessagesView) once the JoinedRoom is available.
 *
 * Default empty string. The badge skips API dispatch when the room id is
 * empty so previews and detached composables are no-ops.
 */
val LocalSirenbertRoomId = compositionLocalOf { "" }
