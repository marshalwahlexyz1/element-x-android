/*
 * Copyright (c) 2025 Element Creations Ltd.
 *
 * SPDX-License-Identifier: AGPL-3.0-only OR LicenseRef-Element-Commercial.
 */

package io.element.android.features.messages.impl.sirenbert

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import android.widget.Toast
import timber.log.Timber

/**
 * Dev-only broadcast hook. Trigger from PowerShell:
 *
 *   adb shell am broadcast -a io.element.android.x.debug.SIRENBERT_RESET
 *
 * Effect:
 *   1. Wipes the in-memory SirenbertCache so all visible badges go back
 *      to "..." and will re-classify on next composition.
 *   2. Bumps the SirenbertSession counter so subsequent POSTs use a fresh
 *      conversation_id (`${roomId}#s${N}`), causing the FastAPI server to
 *      rebuild Stage 2 hidden state from scratch.
 *   3. Shows a Toast confirming the new session counter.
 *
 * Registered in AndroidManifest with android:exported="true" so adb can
 * reach it without granting any new permissions. Receiver only does
 * anything when the SIRENBERT feature flag would have applied; the work
 * is harmless if the flag is off.
 */
class SirenbertResetReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        if (intent.action != ACTION_RESET) return
        SirenbertCache.clear()
        // If the on-device engine is initialised, wipe its per-room buffers
        // too. If it's not (API-only mode), this is a no-op.
        OnDeviceSirenbertEngine.getOrNull(context.applicationContext)?.reset()
        val newCounter = SirenbertSession.bump()
        val message = "SIRENBERT reset (session #$newCounter)"
        Timber.tag("SIRENBERT").i(message)
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    companion object {
        const val ACTION_RESET = "io.element.android.x.debug.SIRENBERT_RESET"
    }
}
