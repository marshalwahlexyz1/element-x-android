/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import androidx.compose.runtime.compositionLocalOf

/**
 * CompositionLocal carrying the current value of
 * `FeatureFlags.SirenbertOnDevice` down to the SIRENBERT badge so it can
 * dispatch to either the on-device engine or the FastAPI client.
 *
 * Defaults to false (API dispatch). Set in MessagesView from
 * TimelineState.displaySirenbertOnDevice.
 */
val LocalSirenbertOnDevice = compositionLocalOf { false }
