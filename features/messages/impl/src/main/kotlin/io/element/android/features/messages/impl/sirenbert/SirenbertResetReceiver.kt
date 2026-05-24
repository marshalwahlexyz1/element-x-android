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
import kotlinx.coroutines.runBlocking
import kotlinx.coroutines.flow.first
import org.json.JSONArray
import org.json.JSONObject
import timber.log.Timber
import java.io.File
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale

/**
 * Dev-only broadcast hook. Two actions handled:
 *
 *   adb shell am broadcast -a io.element.android.x.debug.SIRENBERT_RESET
 *   adb shell am broadcast -a io.element.android.x.debug.SIRENBERT_EXPORT
 *
 * RESET effect:
 *   1. Wipes the in-memory SirenbertCache so all visible badges go back
 *      to "..." and will re-classify on next composition.
 *   2. Bumps the SirenbertSession counter so subsequent POSTs use a fresh
 *      conversation_id (`${roomId}#s${N}`), causing the FastAPI server
 *      and the on-device engine to rebuild Stage 2 hidden state.
 *   3. If the on-device engine exists, wipes its per-room buffers.
 *   4. Toast confirms the new session counter.
 *
 * EXPORT effect:
 *   1. Dumps the current SirenbertCache.results map to a JSON file in
 *      the app's external files dir (adb-pullable, no permission needed).
 *   2. Filename: sirenbert_export_${timestamp}_session${N}.json
 *   3. Toast confirms the path.
 *   4. Logs the absolute path for adb pull.
 *
 * Registered in AndroidManifest with android:exported="true" so adb can
 * reach it without granting any new permissions.
 */
class SirenbertResetReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            ACTION_RESET -> doReset(context)
            ACTION_EXPORT -> doExport(context)
            ACTION_BULK_REPLAY -> doBulkReplay(context)
        }
    }

    private fun doBulkReplay(context: Context) {
        // Run inside a foreground service so the full-test-set run is not
        // suspended/killed when the app is backgrounded. The single-run guard
        // in SirenbertBulkReplay ignores re-triggers while a run is in flight.
        SirenbertBulkReplayService.start(context.applicationContext)
        Toast.makeText(
            context,
            "SIRENBERT bulk replay started (foreground service); watch logcat",
            Toast.LENGTH_SHORT,
        ).show()
    }

    private fun doReset(context: Context) {
        SirenbertCache.clear()
        OnDeviceSirenbertEngine.getOrNull(context.applicationContext)?.reset()
        val newCounter = SirenbertSession.bump()
        val message = "SIRENBERT reset (session #$newCounter)"
        Timber.tag("SIRENBERT").i(message)
        Toast.makeText(context, message, Toast.LENGTH_SHORT).show()
    }

    private fun doExport(context: Context) {
        val results = runBlocking { SirenbertCache.results.first() }
        val sessionN = SirenbertSession.current
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val dir = context.applicationContext.getExternalFilesDir(null)
        if (dir == null) {
            Timber.tag("SIRENBERT").w("EXPORT: getExternalFilesDir(null) returned null")
            Toast.makeText(context, "SIRENBERT export failed: no external dir", Toast.LENGTH_LONG).show()
            return
        }
        if (!dir.exists()) dir.mkdirs()
        val outFile = File(dir, "sirenbert_export_${timestamp}_session${sessionN}.json")

        // Build a stable JSON shape. We do NOT include message bodies; only
        // ids/role/labels/probs so the export is safe to share for
        // equivalence analysis without leaking message content.
        val rows = JSONArray()
        for ((eventId, r) in results) {
            val obj = JSONObject().apply {
                put("event_id", eventId)
                put("role", r.role)
                put("status", r.status.name)
                if (r.verdict != null) put("verdict", r.verdict)
                if (r.trigger != null) put("trigger", r.trigger)
                if (r.state != null) put("state", r.state)
                if (r.suspProb != null) put("susp_prob", r.suspProb.toDouble())
                if (r.scamProb != null) put("scam_prob", r.scamProb.toDouble())
                if (r.errorMessage != null) put("error", r.errorMessage)
            }
            rows.put(obj)
        }
        val header = JSONObject().apply {
            put("schema", "sirenbert.export.v1")
            put("exported_at", timestamp)
            put("session", sessionN)
            put("n_entries", results.size)
        }
        val root = JSONObject().apply {
            put("header", header)
            put("entries", rows)
        }
        outFile.writeText(root.toString(2))
        val msg = "SIRENBERT export -> ${outFile.absolutePath} (${results.size} entries)"
        Timber.tag("SIRENBERT").i(msg)
        Toast.makeText(context, "SIRENBERT exported ${results.size} entries", Toast.LENGTH_LONG).show()
    }

    companion object {
        const val ACTION_RESET = "io.element.android.x.debug.SIRENBERT_RESET"
        const val ACTION_EXPORT = "io.element.android.x.debug.SIRENBERT_EXPORT"
        const val ACTION_BULK_REPLAY = "io.element.android.x.debug.SIRENBERT_BULK_REPLAY"
    }
}
